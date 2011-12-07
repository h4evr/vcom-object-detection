#ifndef __TIMER_H__
#define __TIMER_H__

#include <ctime>

class Timer {
    public:
        Timer();
        void Start();
        void Stop();
        void Reset();
        float GetElapsedTime();
    private:
        struct timespec start;
        struct timespec end;
};

#endif

