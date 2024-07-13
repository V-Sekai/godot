# libTimekeeper

A simple, header-only C++ hierarchical stopwatch system with minimal overhead.

It captures wall-clock, user and system times, as well as a count of how often a stopwatch was resumed.


## Usage example
A full example is available in the [`examples/`](examples/) subdirectory.

### Code

```
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
```
- A `HierarchicalStopWatch` can have dependent child stopwatches. This hierarchy is defined at compile-time, not run-time.
- A `ScopedStopwatch` is a RAII helper that will `resume()` the given stopwatch when constructed and `stop()` it on destruction.

In conjunction with the excellent [nlohmann/json](https://github.com/nlohmann/json) library, libTimekeeper
can also serialize the recorded timing hierarchy to json. You will have to include that library
in your build process and `#include <libTimekeeper/json.h>`, then you can use something like
`nlohmann::json j = HierarchicalStopWatchResult(sw_main)`.

### CMake

You can use `add_subdirectory` on this project, it will provide the `Timekeeper::libTimekeeper` target.

### Output

The above example will output something like this, with pretty boring runtimes:
```
-> % ./examples/example 
                   Wall  User  System  count
main               0 ms  0 ms    0 ms      1
├ foo              0 ms  0 ms    0 ms      2
├ bar              0 ms  0 ms    0 ms      1
│ ├ baz            0 ms  0 ms    0 ms      1
│ ╰ (unaccounted)  0 ms  0 ms    0 ms       
╰ (unaccounted)    0 ms  0 ms    0 ms
```

## License

libTimekeeper is available under the terms of the [MIT License](LICENSE).

## Contact

Author: Martin Heistermann <martin.heistermann@unibe.ch>
