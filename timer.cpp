#include "timer.h"

Timer::Timer() {
    this->Reset();
}

void Timer::Start() {
    clock_gettime(CLOCK_MONOTONIC, &start);
}

void Timer::Stop() {
    clock_gettime(CLOCK_MONOTONIC, &end);
}

void Timer::Reset() {
    clock_gettime(CLOCK_MONOTONIC, &start);
    end.tv_sec = start.tv_sec;
    end.tv_nsec = start.tv_nsec;
}

float Timer::GetElapsedTime() {
    return ((float)end.tv_sec + (float)end.tv_nsec / 1000000000.0f) - 
           ((float)start.tv_sec + (float)start.tv_nsec / 1000000000.0f);
}

