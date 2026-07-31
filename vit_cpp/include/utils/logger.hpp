#pragma once
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace vit::utils {

enum class LogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

class Logger {
public:
    explicit Logger(LogLevel min_level = LogLevel::INFO,
                    const std::string& log_file = "");

    void debug(const std::string& msg);
    void info(const std::string& msg);
    void warn(const std::string& msg);
    void error(const std::string& msg);

    void log_epoch(int epoch, int total_epochs, float loss, float acc, double elapsed_s);
    void log_batch(int batch, int total_batches, float loss, float acc);
    void log_test(float loss, float acc);
    void separator();

    void set_level(LogLevel lvl) { min_level_ = lvl; }

private:
    LogLevel min_level_;
    std::ofstream file_;

    void log(LogLevel lvl, const std::string& msg);
    static std::string level_str(LogLevel lvl);
    static std::string timestamp();
};

Logger& get_logger();

} // namespace vit::utils
