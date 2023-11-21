#pragma once

#include <string>

class Args {
  public:
    int DEFAULT_STEPS = 100;
    double DEFAULT_GRAVITY = 0.0001;
    double DEFAULT_THRESHOLD = 1.0;
    double DEFAULT_TIMESTEP = 0.005;
    double DEFAULT_RLIMIT = 0.03;
    bool DEFAULT_VISUAL = false;
    bool DEFAULT_SEQUENTIAL = false;

    Args(int argc, char **argv);

    std::string input() const;
    std::string output() const;
    int steps() const;
    double gravity() const;
    double threshold() const;
    double timestep() const;
    double rlimit() const;
    bool visual() const;
    bool sequential() const;

  private:
    std::string input_;
    std::string output_{"/dev/stdout"};
    int steps_{DEFAULT_STEPS};
    double gravity_{DEFAULT_GRAVITY};
    double threshold_{DEFAULT_THRESHOLD};
    double timestep_{DEFAULT_TIMESTEP};
    double rlimit_{DEFAULT_RLIMIT};
    bool visual_{DEFAULT_VISUAL};
    bool sequential_{DEFAULT_SEQUENTIAL};

    void check_args() const;
};
