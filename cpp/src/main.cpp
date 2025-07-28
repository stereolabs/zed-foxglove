#include <atomic>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <signal.h>
#include <thread>
#include <vector>

#include <sl/Camera.hpp>

#include <foxglove/context.hpp>
#include <foxglove/error.hpp>
#include <foxglove/foxglove.hpp>
#include <foxglove/mcap.hpp>
#include <foxglove/schemas.hpp>
#include <foxglove/server.hpp>

// Global variables
std::vector<std::unique_ptr<sl::Camera>> zed_list;
std::vector<sl::Mat> left_list;
std::vector<sl::Mat> depth_list;
std::vector<uint64_t> timestamp_list;
std::vector<std::thread> thread_list;
std::atomic<bool> stop_signal {false};

// Command line arguments
struct Args {
    std::string mcap_file = "";
    bool enable_ws = false;

    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--mcap" && i + 1 < argc) {
                mcap_file = argv[++i];
            } else if (arg == "--ws") {
                enable_ws = true;
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [options]\n";
                std::cout << "  --mcap <file>  Output MCAP file name (default: "
                             "output.mcap)\n";
                std::cout << "  --ws           Enable WebSocket server\n";
                std::cout << "  --help, -h     Show this help message\n";
                exit(0);
            }
        }
    }
};

// Signal handler for graceful shutdown
void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down gracefully...\n";
    stop_signal = true;
}

// Function to encode image as JPEG
std::vector<std::byte> encodeJpeg(const cv::Mat& image, int quality = 90) {
    std::vector<std::uint8_t> buffer;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
    cv::imencode(".jpg", image, buffer, params);

    // Convert uint8_t to std::byte using reinterpret_cast
    const std::byte* byte_ptr = reinterpret_cast<const std::byte*>(buffer.data());
    return std::vector<std::byte>(byte_ptr, byte_ptr + buffer.size());
}

// Convert sl::Mat to cv::Mat
cv::Mat slMatToCvMat(const sl::Mat& input) {
    int cv_type = -1;
    switch (input.getDataType()) {
        case sl::MAT_TYPE::F32_C1:
            cv_type = CV_32FC1;
            break;
        case sl::MAT_TYPE::F32_C2:
            cv_type = CV_32FC2;
            break;
        case sl::MAT_TYPE::F32_C3:
            cv_type = CV_32FC3;
            break;
        case sl::MAT_TYPE::F32_C4:
            cv_type = CV_32FC4;
            break;
        case sl::MAT_TYPE::U8_C1:
            cv_type = CV_8UC1;
            break;
        case sl::MAT_TYPE::U8_C2:
            cv_type = CV_8UC2;
            break;
        case sl::MAT_TYPE::U8_C3:
            cv_type = CV_8UC3;
            break;
        case sl::MAT_TYPE::U8_C4:
            cv_type = CV_8UC4;
            break;
        default:
            break;
    }

    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>());
}

foxglove::schemas::PointCloud makePointCloud(int zed_id, const sl::Mat& point_cloud) {
    const std::byte* byte_ptr = reinterpret_cast<const std::byte*>(point_cloud.getPtr<sl::uchar1>());
    size_t data_size = point_cloud.getWidth() * point_cloud.getHeight() * point_cloud.getPixelBytes();

    // Create vector from the data
    std::vector<std::byte> point_cloud_data(byte_ptr, byte_ptr + data_size);

    foxglove::schemas::PointCloud point_cloud_msg;
    point_cloud_msg.data = point_cloud_data;
    point_cloud_msg.frame_id = "zed_" + std::to_string(zed_id);
    point_cloud_msg.pose = foxglove::schemas::Pose {
        .position = foxglove::schemas::Vector3 {0.0, 0.0, 0.0},
        .orientation = foxglove::schemas::Quaternion {0.0, 0.0, 0.0, 1.0}  // Identity quaternion
    };
    point_cloud_msg.point_stride = 16;  // 4 floats (x, y, z) + 4 bytes (rgba)
    point_cloud_msg.fields = {
        foxglove::schemas::PackedElementField {
                                               .name = "x",
                                               .offset = 0,
                                               .type = foxglove::schemas::PackedElementField::NumericType::FLOAT32,
                                               },
        foxglove::schemas::PackedElementField {
                                               .name = "y",
                                               .offset = 4,
                                               .type = foxglove::schemas::PackedElementField::NumericType::FLOAT32,
                                               },
        foxglove::schemas::PackedElementField {
                                               .name = "z",
                                               .offset = 8,
                                               .type = foxglove::schemas::PackedElementField::NumericType::FLOAT32,
                                               },
        foxglove::schemas::PackedElementField {
                                               .name = "red",
                                               .offset = 12,
                                               .type = foxglove::schemas::PackedElementField::NumericType::UINT8,
                                               },
        foxglove::schemas::PackedElementField {
                                               .name = "green",
                                               .offset = 13,
                                               .type = foxglove::schemas::PackedElementField::NumericType::UINT8,
                                               },
        foxglove::schemas::PackedElementField {
                                               .name = "blue",
                                               .offset = 14,
                                               .type = foxglove::schemas::PackedElementField::NumericType::UINT8,
                                               },
        foxglove::schemas::PackedElementField {
                                               .name = "alpha",
                                               .offset = 15,
                                               .type = foxglove::schemas::PackedElementField::NumericType::UINT8,
                                               },
    };

    return point_cloud_msg;
}

// Camera capture function for each thread
void grab_run(int index) {

    sl::RuntimeParameters runtime_params;

    auto left_channel = foxglove::schemas::CompressedImageChannel::create("image_" + std::to_string(index)).value();
    auto depth_channel = foxglove::schemas::CompressedImageChannel::create("depth_" + std::to_string(index)).value();
    auto point_cloud_channel = foxglove::schemas::PointCloudChannel::create("point_cloud_" + std::to_string(index)).value();

    while (!stop_signal) {
        sl::ERROR_CODE err = zed_list[index]->grab(runtime_params);

        if (err == sl::ERROR_CODE::SUCCESS) {
            // Retrieve images
            zed_list[index]->retrieveImage(left_list[index], sl::VIEW::LEFT);
            zed_list[index]->retrieveImage(depth_list[index], sl::VIEW::DEPTH);

            // Convert to OpenCV format
            cv::Mat left_cv = slMatToCvMat(left_list[index]);
            cv::Mat depth_cv = slMatToCvMat(depth_list[index]);

            // Encode as JPEG
            auto left_jpeg = encodeJpeg(left_cv, 90);
            auto depth_jpeg = encodeJpeg(depth_cv, 90);

            // Get timestamp
            timestamp_list[index] = zed_list[index]->getTimestamp(sl::TIME_REFERENCE::CURRENT).data_ns;

            // Create compressed image data
            foxglove::schemas::CompressedImage left_msg;
            left_msg.data = left_jpeg;
            left_msg.format = "jpeg";

            foxglove::schemas::CompressedImage depth_msg;
            depth_msg.data = depth_jpeg;
            depth_msg.format = "jpeg";

            left_channel.log(left_msg);
            depth_channel.log(depth_msg);

            // Retrieve point cloud
            sl::Mat point_cloud;
            zed_list[index]->retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA);

            auto point_cloud_msg = makePointCloud(index, point_cloud);
            point_cloud_channel.log(point_cloud_msg);
        }

        // Small delay to prevent excessive CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Close camera
    zed_list[index]->close();
}

int main(int argc, char* argv[]) {
    Args args;
    args.parse(argc, argv);

    // Set up signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Set log level
    foxglove::setLogLevel(foxglove::LogLevel::Debug);

    std::cout << "ZED to Foxglove Streaming Application\n";
    std::cout << "MCAP file: " << args.mcap_file << std::endl;
    std::cout << "WebSocket server: " << (args.enable_ws ? "enabled" : "disabled") << std::endl;

    // Initialize Foxglove WebSocket server

    foxglove::WebSocketServerOptions options = {};
    options.name = "zed-foxglove-server";
    options.host = "127.0.0.1";
    options.port = 8765;
    options.capabilities = foxglove::WebSocketServerCapabilities::ClientPublish;
    options.supported_encodings = {"json"};

    std::shared_ptr<foxglove::WebSocketServer> server = nullptr;
    if (args.enable_ws) {

        auto serverResult = foxglove::WebSocketServer::create(std::move(options));
        if (!serverResult.has_value()) {
            std::cerr << "Failed to create server: " << foxglove::strerror(serverResult.error()) << std::endl;
            return 1;
        }
        server = std::make_shared<foxglove::WebSocketServer>(std::move(serverResult.value()));
        if (!server) {
            std::cerr << "Failed to create WebSocket server instance." << std::endl;
            return 1;
        }
        std::cout << "WebSocket server started on port " << options.port << std::endl;
    }

    // Initialize MCAP writer
    std::shared_ptr<foxglove::McapWriter> mcap_writer = nullptr;
    if (!args.mcap_file.empty()) {
        foxglove::McapWriterOptions mcap_options;
        mcap_options.path = args.mcap_file;

        auto writerResult = foxglove::McapWriter::create(mcap_options);
        if (!writerResult.has_value()) {
            std::cerr << "Failed to create MCAP writer: " << foxglove::strerror(writerResult.error()) << std::endl;
            return 1;
        }
        mcap_writer = std::make_shared<foxglove::McapWriter>(std::move(writerResult.value()));
        std::cout << "MCAP writer initialized: " << args.mcap_file << std::endl;
    }

    // Initialize ZED SDK
    std::cout << "Initializing ZED cameras..." << std::endl;

    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION::AUTO;
    init_params.camera_fps = 30;
    init_params.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP;
    init_params.coordinate_units = sl::UNIT::METER;

    // List and open cameras
    std::vector<sl::DeviceProperties> cameras = sl::Camera::getDeviceList();
    std::cout << "Found " << cameras.size() << " ZED camera(s)" << std::endl;

    for (size_t i = 0; i < cameras.size(); ++i) {
        init_params.input.setFromSerialNumber(cameras[i].serial_number);

        std::cout << "Opening ZED " << cameras[i].serial_number << std::endl;

        auto zed = std::make_unique<sl::Camera>();
        left_list.emplace_back();
        depth_list.emplace_back();
        timestamp_list.push_back(0);

        sl::ERROR_CODE status = zed->open(init_params);
        if (status != sl::ERROR_CODE::SUCCESS) {
            std::cerr << "Failed to open camera " << cameras[i].serial_number << ": " << sl::toString(status) << std::endl;
            continue;
        }

        zed_list.push_back(std::move(zed));
    }

    if (zed_list.empty()) {
        std::cerr << "No cameras opened successfully. Exiting." << std::endl;
        return 1;
    }

    // Start camera threads
    std::cout << "Starting camera capture threads..." << std::endl;
    for (size_t i = 0; i < zed_list.size(); ++i) {
        thread_list.emplace_back(grab_run, i);
    }

    std::cout << "Running... Press Ctrl+C to stop." << std::endl;

    // Main loop
    try {
        while (!stop_signal) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception in main loop: " << e.what() << std::endl;
    }

    // Cleanup
    std::cout << "Shutting down..." << std::endl;
    stop_signal = true;

    // Wait for all threads to finish
    for (auto& thread : thread_list) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Stop WebSocket server
    if (server) {
        server->stop();
    }

    // Close MCAP writer
    if (mcap_writer) {
        mcap_writer->close();
    }

    std::cout << "Shutdown complete." << std::endl;
    return 0;
}
