#include <libTimekeeper/StopWatch.hh>
#include <libTimekeeper/StopWatchPrinting.hh>
#include <iostream>

using HSW = Timekeeper::HierarchicalStopWatch;
using Timekeeper::ScopedStopWatch;

HSW sw_main("main");
HSW sw_foo("foo", sw_main);
HSW sw_bar("bar", sw_main);
HSW sw_baz("baz", sw_bar);

void foo() {
    ScopedStopWatch _{sw_foo};
}
void baz() {
    ScopedStopWatch _{sw_baz};
}
void bar() {
    ScopedStopWatch _{sw_bar};
    baz();
}
void work() {
    ScopedStopWatch _{sw_main};
    foo();
    bar();
    foo();
}
int main() {
    work();
    std::cout << sw_main << std::endl;

}
