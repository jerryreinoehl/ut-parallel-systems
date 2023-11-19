#pragma once

#include <string>

class Args {
  public:
    int DEFAULT_STEPS = 100;
    double DEFAULT_GRAVITY = 0.0001;
    double DEFAULT_THRESHOLD = 1.0;
    double DEFAULT_TIMESTEP = 0.005;
    bool DEFAULT_VISUAL = false;

    Args(int argc, char **argv);

    std::string input() const;
    std::string output() const;
    int steps() const;
    double gravity() const;
    double threshold() const;
    double timestep() const;
    bool visual() const;

  private:
    std::string input_;
    std::string output_;
    int steps_{DEFAULT_STEPS};
    double gravity_{DEFAULT_GRAVITY};
    double threshold_{DEFAULT_THRESHOLD};
    double timestep_{DEFAULT_TIMESTEP};
    bool visual_{DEFAULT_VISUAL};

    void check_args() const;
};
