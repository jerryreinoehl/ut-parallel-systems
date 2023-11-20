#include "args.h"

#include <filesystem>
#include <iostream>

Args::Args(int argc, char **argv) {
  std::string arg;
  for (char **it = &argv[0], **end = &argv[argc]; it != end; ++it) {
    arg = std::string{*it};

    if (arg == "-i") {
      input_ = *(++it);
    } else if (arg == "-o") {
      output_ = *(++it);
    } else if (arg == "-g") {
      gravity_ = std::stod(*(++it));
    } else if (arg == "-s") {
      steps_ = std::stoi(*(++it));
    } else if (arg == "-t") {
      threshold_ = std::stod(*(++it));
    } else if (arg == "-d") {
      timestep_ = std::stod(*(++it));
    } else if (arg == "-r") {
      rlimit_ = std::stod(*(++it));
    } else if (arg == "-V") {
      visual_ = true;
    }
  }

  check_args();
}

std::string Args::input() const {
  return input_;
}

std::string Args::output() const {
  return output_;
}

int Args::steps() const {
  return steps_;
}

double Args::gravity() const {
  return gravity_;
}

double Args::threshold() const {
  return threshold_;
}

double Args::timestep() const {
  return timestep_;
}

double Args::rlimit() const {
  return rlimit_;
}

bool Args::visual() const {
  return visual_;
}

void Args::check_args() const {
  // Check if input file exists.
  if (!std::filesystem::exists(input_)) {
    std::cerr << "Input file doesn't exist.\n";
    exit(1);
  }

  // Ensure an output file is given.
  if (output_ == "") {
    std::cerr << "Output file not specified.\n";
    exit(1);
  }
}
