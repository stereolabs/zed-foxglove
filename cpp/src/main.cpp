#include <foxglove/foxglove.hpp>
#include <foxglove/context.hpp>
#include <foxglove/error.hpp>
#include <foxglove/mcap.hpp>

#include <iostream>

int main(int argc, const char* argv[]) {
  // This doesn't affect what gets logged to the MCAP file, this is for troubleshooting the SDK integration
  foxglove::setLogLevel(foxglove::LogLevel::Debug);

  foxglove::McapWriterOptions mcap_options = {};
  mcap_options.path = "quickstart-cpp.mcap";
  auto writerResult = foxglove::McapWriter::create(mcap_options);
  if (!writerResult.has_value()) {
    std::cerr << "Failed to create writer: " << foxglove::strerror(writerResult.error()) << '\n';
    return 1;
  }
  auto writer = std::move(writerResult.value());

  return 0;
}