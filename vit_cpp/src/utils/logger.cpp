#include "utils/logger.hpp"
#include <iostream>

namespace vit::utils {

Logger::Logger(LogLevel min_level, const std::string& log_file)
    : min_level_(min_level) {
    if (!log_file.empty())
        file_.open(log_file, std::ios::app);
}

void Logger::debug(const std::string& msg) { log(LogLevel::DEBUG, msg); }
void Logger::info (const std::string& msg) { log(LogLevel::INFO,  msg); }
void Logger::warn (const std::string& msg) { log(LogLevel::WARN,  msg); }
void Logger::error(const std::string& msg) { log(LogLevel::ERROR, msg); }

void Logger::log_epoch(int epoch, int total, float loss, float acc, double elapsed) {
    std::ostringstream oss;
    oss << "Epoca [" << std::setw(2) << epoch << "/" << total << "]"
        << "  perdida=" << std::fixed << std::setprecision(4) << loss
        << "  exactitud=" << std::fixed << std::setprecision(2) << acc * 100.f << "%"
        << "  tiempo=" << std::fixed << std::setprecision(1) << elapsed << "s";
    info(oss.str());
}

void Logger::log_batch(int batch, int total, float loss, float acc) {
    std::ostringstream oss;
    oss << "  lote [" << std::setw(4) << batch << "/" << total << "]"
        << "  perdida=" << std::fixed << std::setprecision(4) << loss
        << "  exactitud=" << std::fixed << std::setprecision(1) << acc * 100.f << "%";
    debug(oss.str());
}

void Logger::log_test(float loss, float acc) {
    std::ostringstream oss;
    oss << "==> Prueba  perdida=" << std::fixed << std::setprecision(4) << loss
        << "  exactitud=" << std::fixed << std::setprecision(2) << acc * 100.f << "%";
    info(oss.str());
}

void Logger::separator() {
    info(std::string(60, '-'));
}

void Logger::log(LogLevel lvl, const std::string& msg) {
    if (lvl < min_level_) return;
    std::string line = "[" + timestamp() + "] [" + level_str(lvl) + "] " + msg;
    std::cout << line << "\n";
    if (file_.is_open()) file_ << line << "\n";
}

std::string Logger::level_str(LogLevel lvl) {
    switch (lvl) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return " INFO";
        case LogLevel::WARN:  return " AVISO";
        case LogLevel::ERROR: return "ERROR";
    }
    return "?";
}

std::string Logger::timestamp() {
    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                   now.time_since_epoch()) % 1000;
    std::ostringstream oss;
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &tt);
#else
    localtime_r(&tt, &tm_buf);
#endif
    oss << std::put_time(&tm_buf, "%H:%M:%S")
        << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

Logger& get_logger() {
    static Logger instance(LogLevel::INFO);
    return instance;
}

} // namespace vit::utils
