# C++ Primer for Python Developers

This document explains the C++ concepts you need to read and understand this codebase, using Python as a starting point. Every example points to a real file in this repo.

---

## Table of Contents

1. [The Big Picture: How C++ Differs from Python](#1-the-big-picture)
2. [#include and Namespaces](#2-include-and-namespaces)
3. [Types and static_cast](#3-types-and-static_cast)
4. [const and constexpr](#4-const-and-constexpr)
5. [std::vector and Other Containers](#5-stdvector-and-other-containers)
6. [std::span — Views Without Copying](#6-stdspan--views-without-copying)
7. [Smart Pointers — Memory Without Leaks](#7-smart-pointers--memory-without-leaks)
8. [Move Semantics — std::move](#8-move-semantics--stdmove)
9. [Lambdas](#9-lambdas)
10. [Virtual Functions and Inheritance](#10-virtual-functions-and-inheritance)
11. [Templates — Generic Code](#11-templates--generic-code)
12. [RAII — The "with" Statement Baked Into the Language](#12-raii--the-with-statement-baked-into-the-language)
13. [std::atomic — Thread-Safe Variables](#13-stdatomic--thread-safe-variables)
14. [enum class — Type-Safe Enumerations](#14-enum-class--type-safe-enumerations)
15. [Structured Bindings — Tuple Unpacking](#15-structured-bindings--tuple-unpacking)
16. [Exceptions](#16-exceptions)
17. [Standard Algorithms](#17-standard-algorithms)
18. [nanobind — C++ ↔ Python Bridge](#18-nanobind--c--python-bridge)
19. [CMake — The Build System](#19-cmake--the-build-system)
20. [How the Solver Fits Together](#20-how-the-solver-fits-together)
21. [Quick Reference Cheat Sheet](#21-quick-reference-cheat-sheet)

---

## 1. The Big Picture

| | Python | C++ |
|---|---|---|
| **Execution** | Interpreted at runtime | Compiled to native machine code |
| **Typing** | Dynamic — types checked at runtime | Static — types checked at compile time |
| **Memory** | Garbage collector frees objects | You (or smart pointers) manage lifetimes |
| **Speed** | Slower; great for prototyping | Faster; used for performance-critical work |
| **File structure** | One `.py` file per module | Two files per module: `.h` (declaration) and `.cpp` (implementation) |

### Header files vs. implementation files

Python has one file per module. C++ splits each module into two:

- **`.h` (header)** — declares *what* exists: class names, method signatures, constants. Think of it as the "interface" or "type stub".
- **`.cpp` (implementation)** — defines *how* it works: the actual code.

```
# Python: one file
# separator.py
class Separator:
    def name(self) -> str: ...
    def separate(self, ctx) -> list[Cut]: ...
```

```cpp
// C++: two files
// separator.h — declares the interface
class Separator {
 public:
  virtual std::string name() const = 0;
  virtual std::vector<Cut> separate(const SeparationContext& ctx) = 0;
};

// sec_separator.cpp — implements the behaviour
std::vector<Cut> SECSeparator::separate(const SeparationContext& ctx) {
  // actual code here
}
```

See [src/sep/separator.h](src/sep/separator.h) and [src/sep/sec_separator.h](src/sep/sec_separator.h).

`#pragma once` at the top of every header is a guard that prevents the file from being included twice — equivalent to Python's import caching.

---

## 2. #include and Namespaces

### #include

`#include` is like Python's `import`. It pulls in declarations from another file.

```python
# Python
from sep.separator import Separator
```

```cpp
// C++
#include "sep/separator.h"
```

Angle brackets `<vector>` are for standard library headers; quotes `"sep/separator.h"` are for project files.

### Namespaces

Namespaces are like Python packages — they group related names to avoid collisions.

```python
# Python
import cptp.sep
cptp.sep.Separator
```

```cpp
// C++
namespace cptp::sep {   // declared at top of file
  class Separator { ... };
}

// used elsewhere as:
cptp::sep::Separator
```

Every source file in this repo opens a namespace at the top. For example [src/sep/separator.h:10](src/sep/separator.h#L10):

```cpp
namespace cptp::sep {
```

And [src/parallel/parallel.h:7](src/parallel/parallel.h#L7):

```cpp
namespace cptp::parallel {
```

---

## 3. Types and static_cast

### Static types

Python lets you assign any value to any variable. C++ requires you to declare the type upfront, and the compiler enforces it.

```python
# Python — works fine
x = 42
x = "hello"  # reassign to different type
```

```cpp
// C++ — type is fixed at declaration
int32_t x = 42;
// x = "hello";  // compile error
```

Common types you'll see:
- `int32_t` / `int64_t` — fixed-size integers (32 or 64 bits)
- `double` — 64-bit floating point (Python's `float`)
- `bool` — true/false
- `std::string` — text (Python's `str`)
- `void` — return type for functions that return nothing

### auto

`auto` tells the compiler to infer the type — like Python's duck typing, but checked at compile time:

```cpp
auto x = 42;          // compiler knows this is int
auto s = std::string{"hello"};  // compiler knows this is std::string
```

### static_cast

`static_cast<T>(x)` is an explicit type conversion. It's the safe version of Python's `int(x)` or `float(x)`.

```python
# Python
n = int(3.7)  # → 3
```

```cpp
// C++
int32_t n = static_cast<int32_t>(3.7);  // → 3
```

You'll see this in [src/parallel/parallel.h:16](src/parallel/parallel.h#L16) converting a `uint` to `int`:

```cpp
num_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
```

---

## 4. const and constexpr

### const — "I promise not to change this"

`const` is a compile-time guarantee that a value won't be modified. Python has no direct equivalent (though `Final` hints at it).

```python
# Python — only a hint, not enforced
from typing import Final
MAX: Final = 100
```

```cpp
// C++ — enforced by the compiler
const int32_t max = 100;
// max = 200;  // compile error
```

`const` on a method means the method doesn't modify the object:

```cpp
bool is_optimal() const { return status == Status::Optimal; }
//                ^^^^^ this method cannot change any member variables
```

See [src/core/solution.h:43](src/core/solution.h#L43).

### const& — Read without copying

When a function takes `const SomeType& x`, it reads the object without making a copy. This is how Python function arguments work by default — but in C++ you must say it explicitly.

```python
# Python — always passes by reference
def process(problem: Problem) -> None: ...
```

```cpp
// C++ — const& = read-only reference, no copy
void process(const Problem& problem) { ... }
```

### constexpr — Known at compile time

`constexpr` values are computed at compile time — zero runtime cost.

```cpp
// src/sep/separator.h:14
constexpr double kInfiniteCapacity = 1e17;

// src/heuristic/primal_heuristic.h:23
inline constexpr int32_t kDefaultLsMaxIterPerStart = 1000;
```

These are like Python module-level constants, but the compiler substitutes them directly into the binary.

---

## 5. std::vector and Other Containers

### std::vector — Python's list

`std::vector<T>` is a dynamically-sized array, like Python's `list`, but every element must be the same type.

```python
# Python
numbers: list[int] = [1, 2, 3]
numbers.append(4)
```

```cpp
// C++
std::vector<int32_t> numbers = {1, 2, 3};
numbers.push_back(4);       // like append()
numbers.emplace_back(5);    // constructs in place (more efficient)
numbers.reserve(100);       // pre-allocate capacity (no Python equivalent needed)
```

In [src/core/digraph.h:38-43](src/core/digraph.h#L38-L43) you can see vectors used to store graph data:

```cpp
std::vector<int32_t> sources_;
std::vector<int32_t> targets_;
std::vector<int32_t> out_begin_;
std::vector<int32_t> out_arcs_;
```

### std::map and std::unordered_map — Python's dict

```python
# Python
stats: dict[str, int] = {"SEC": 5, "RCI": 3}
```

```cpp
// std::map — sorted by key (like a sorted dict)
std::map<std::string, SeparatorStats> separator_stats;

// std::unordered_map — hash map (like Python dict)
std::unordered_map<std::string, int32_t> name_to_index;
```

See [src/core/solution.h:39](src/core/solution.h#L39) for the map used to track per-separator statistics.

### Iterating containers

```python
# Python
for item in my_list:
    print(item)
```

```cpp
// C++ range-for (identical concept)
for (const auto& item : my_list) {
    // use item
}
```

---

## 6. std::span — Views Without Copying

`std::span<const T>` is a lightweight view into an existing array or vector. It knows the pointer and the length, but it doesn't own the data and doesn't copy it.

The closest Python analogy is a slice or `memoryview`:

```python
# Python
data = [1, 2, 3, 4, 5]
view = data[1:4]  # creates a new list (copies)
mv = memoryview(bytes(data))  # zero-copy view
```

```cpp
// C++ — zero-copy view, no allocation
std::vector<double> data = {1, 2, 3, 4, 5};
std::span<const double> view = data;  // points into data, no copy
```

In [src/sep/separation_oracle.h:60-62](src/sep/separation_oracle.h#L60-L62), `separate()` accepts spans so the caller's vectors are read directly:

```cpp
std::vector<Cut> separate(std::span<const double> x_values,
                          std::span<const double> y_values,
                          int32_t x_offset, int32_t y_offset) const;
```

In [src/core/digraph.h:22-25](src/core/digraph.h#L22-L25), `out_arcs()` returns a span into the internal storage — no copy:

```cpp
std::span<const int32_t> out_arcs(int32_t v) const {
  return {out_arcs_.data() + out_begin_[v],
          out_arcs_.data() + out_begin_[v + 1]};
}
```

---

## 7. Smart Pointers — Memory Without Leaks

Python objects are automatically freed by the garbage collector. C++ has no GC, but **smart pointers** provide automatic cleanup using RAII (see [section 12](#12-raii--the-with-statement-baked-into-the-language)).

### std::unique_ptr — Single owner

`unique_ptr<T>` owns an object exclusively. When the pointer goes out of scope, the object is automatically deleted. There can only be one owner at a time.

```python
# Python — GC handles this
sep = SECSeparator()
# sep is freed when no references remain
```

```cpp
// C++ — unique_ptr owns and frees the object
std::unique_ptr<Separator> sep = std::make_unique<SECSeparator>();
// sep is freed when it goes out of scope
```

In [src/sep/separation_oracle.h:92](src/sep/separation_oracle.h#L92), separators are stored as unique_ptrs:

```cpp
std::vector<std::unique_ptr<Separator>> separators_;
```

To add a separator, you transfer ownership ([src/sep/separation_oracle.h:40](src/sep/separation_oracle.h#L40)):

```cpp
void add_separator(std::unique_ptr<Separator> sep);
```

### std::shared_ptr — Multiple owners

`shared_ptr<T>` uses reference counting, like Python's objects. Multiple shared_ptrs can point to the same object; the object is freed when the last one dies.

In [src/model/highs_bridge.h](src/model/highs_bridge.h), the interrupt flag is shared between the bridge and callback closures:

```cpp
std::shared_ptr<std::atomic<bool>> interrupt_flag_;
```

### make_unique / make_shared

Always use these factory functions to create smart pointers — they're safer and more efficient than `new`:

```cpp
auto sep = std::make_unique<SECSeparator>();
auto flag = std::make_shared<std::atomic<bool>>(false);
```

---

## 8. Move Semantics — std::move

Normally, assigning a C++ object copies all its data. For large objects (like a vector with a million elements), that's expensive. `std::move` transfers ownership instead of copying — like swapping the internal pointer.

```python
# Python — no equivalent needed; assignment just adds a reference
a = [1, 2, 3, 4, 5]
b = a  # both a and b point to the same list
```

```cpp
// C++ — assignment copies by default
std::vector<int> a = {1, 2, 3, 4, 5};
std::vector<int> b = a;            // copies all data
std::vector<int> c = std::move(a); // transfers: c owns the data, a is now empty
```

In [src/model/model.cpp:403-405](src/model/model.cpp#L403-L405), the entire `Problem` object is moved in rather than copied:

```cpp
void Model::set_problem(Problem prob) {
  problem_ = std::move(prob);
  built_ = true;
}
```

In [src/core/digraph.h:106](src/core/digraph.h#L106), the builder returns both the graph and capacities by moving them out:

```cpp
return {std::move(g), std::move(caps_)};
```

After a `std::move`, the original variable is in a valid but unspecified state — don't use it.

---

## 9. Lambdas

C++ lambdas are like Python lambdas, but more powerful. They can span multiple lines, have explicit capture lists, and be used anywhere a function is expected.

```python
# Python lambda (single expression)
sorter = lambda x: x.violation
items.sort(key=sorter)
```

```cpp
// C++ lambda (can be multi-line)
auto sorter = [](const Cut& a, const Cut& b) {
  return a.violation > b.violation;
};
std::sort(cuts.begin(), cuts.end(), sorter);
```

### Capture lists

The `[...]` at the start is the capture list — it controls what variables from the enclosing scope the lambda can see.

```python
# Python — closures capture everything automatically
threshold = 0.01
filtered = [x for x in cuts if x.violation > threshold]
```

```cpp
// C++ — you must say what to capture
double threshold = 0.01;

// [&] captures everything by reference (most common in this codebase)
auto keep = [&](const Cut& c) { return c.violation > threshold; };

// [threshold] captures just threshold by value
auto keep2 = [threshold](const Cut& c) { return c.violation > threshold; };
```

In [src/parallel/parallel.h:28](src/parallel/parallel.h#L28), a lambda is passed to a thread:

```cpp
threads.emplace_back([&f, lo, hi] {
  for (int i = lo; i < hi; ++i) f(i);
});
```

In [src/util/logger.h:31](src/util/logger.h#L31), a variadic template lambda-like pattern formats a log message:

```cpp
template <typename... Args>
void log(std::format_string<Args...> fmt, Args&&... args) {
  auto msg = std::format(fmt, std::forward<Args>(args)...);
  log(std::string_view{msg});
}
```

---

## 10. Virtual Functions and Inheritance

C++ inheritance works like Python's class inheritance. Virtual functions are like Python's `@abstractmethod`.

### Abstract base class

```python
# Python
from abc import ABC, abstractmethod

class Separator(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def separate(self, ctx: SeparationContext) -> list[Cut]: ...
```

```cpp
// C++ — src/sep/separator.h:17-22
class Separator {
 public:
  virtual ~Separator() = default;         // virtual destructor (always needed)
  virtual std::string name() const = 0;   // = 0 means "must override" (pure virtual)
  virtual std::vector<Cut> separate(const SeparationContext& ctx) = 0;
};
```

### Concrete derived class

```python
# Python
class SECSeparator(Separator):
    def name(self) -> str:
        return "SEC"

    def separate(self, ctx: SeparationContext) -> list[Cut]:
        # real implementation
        ...
```

```cpp
// C++ — src/sep/sec_separator.h:10-14
class SECSeparator : public Separator {      // "public Separator" = inherits from Separator
 public:
  std::string name() const override { return "SEC"; }  // override = "I'm implementing the virtual method"
  std::vector<Cut> separate(const SeparationContext& ctx) override;
};
```

`override` is optional but strongly encouraged — the compiler will error if you misspell the method name or get the signature wrong.

### Polymorphism via pointers

```python
# Python
separators: list[Separator] = [SECSeparator(), RCISeparator()]
for sep in separators:
    cuts = sep.separate(ctx)  # calls the right subclass method
```

```cpp
// C++ — works the same, but requires pointer/reference for polymorphism
std::vector<std::unique_ptr<Separator>> separators;
separators.push_back(std::make_unique<SECSeparator>());

for (const auto& sep : separators) {
    auto cuts = sep->separate(ctx);  // calls the right subclass method
}
```

See [src/sep/separation_oracle.h:92](src/sep/separation_oracle.h#L92) for the actual separator storage.

---

## 11. Templates — Generic Code

Templates let you write code that works for multiple types, determined at compile time. The concept is similar to Python's duck typing or generics, but the compiler generates a separate version for each type used.

```python
# Python — works with any type (duck typing)
def parallel_for(begin: int, end: int, f: Callable[[int], None]) -> None:
    for i in range(begin, end):
        f(i)
```

```cpp
// C++ — template: F can be any callable type
// src/parallel/parallel.h:11-12
template <typename F>
void parallel_for(int begin, int end, F&& f, int num_threads = 0) {
  // launches threads, each calling f(i) for a range of i
}
```

The `typename F` says "F is some type, figure it out from context." When you call `parallel_for(0, 100, my_lambda)`, the compiler substitutes the actual type of `my_lambda` for `F`.

### Variadic templates

`typename... Args` is a variable number of type parameters, like Python's `*args`:

```python
# Python
def log(fmt: str, *args: Any) -> None:
    print(fmt.format(*args))
```

```cpp
// C++ — src/util/logger.h:30-34
template <typename... Args>
void log(std::format_string<Args...> fmt, Args&&... args) {
  auto msg = std::format(fmt, std::forward<Args>(args)...);
  log(std::string_view{msg});
}
```

---

## 12. RAII — The "with" Statement Baked Into the Language

RAII (Resource Acquisition Is Initialization) is the C++ idiom where resources are tied to object lifetimes. When an object goes out of scope, its destructor runs and releases the resource automatically. This is exactly what Python's `with` statement does — but in C++ it happens for *every* object, everywhere.

### Mutex locks

```python
# Python
import threading
lock = threading.Lock()

with lock:
    # critical section
    shared_data += 1
# lock released automatically
```

```cpp
// C++ — std::lock_guard does the same thing
// src/util/logger.h:24
std::lock_guard lock(mu_);  // acquires mu_
// critical section
// lock released automatically when lock goes out of scope
```

### jthread — Threads that join automatically

```python
# Python
import threading
t = threading.Thread(target=worker)
t.start()
t.join()  # you must remember to call join
```

```cpp
// C++ std::jthread joins automatically on destruction
// src/parallel/parallel.h:20,33
std::vector<std::jthread> threads;
threads.emplace_back([&f, lo, hi] {
  for (int i = lo; i < hi; ++i) f(i);
});
// threads are joined when the vector is destroyed (end of function)
```

This is why `task_group` in [src/parallel/parallel.h:37-54](src/parallel/parallel.h#L37-L54) has an empty destructor — the `jthread` objects in `threads_` do the cleanup:

```cpp
~task_group() { wait(); }
```

---

## 13. std::atomic — Thread-Safe Variables

When multiple threads read and write the same variable simultaneously, you get data races (undefined behavior). `std::atomic<T>` makes single-variable operations thread-safe without needing a mutex.

```python
# Python — GIL makes simple assignments thread-safe (most of the time)
counter = 0
counter += 1  # not always safe across threads
```

```cpp
// C++ — atomic makes this safe
std::atomic<int64_t> counter{0};
counter.fetch_add(1);  // atomic increment
counter.load();        // atomic read
counter.store(42);     // atomic write
```

In [src/util/logger.h:38](src/util/logger.h#L38), the enabled flag is atomic so any thread can toggle logging:

```cpp
std::atomic<bool> enabled_{true};
```

In [src/heuristic/primal_heuristic.cpp](src/heuristic/primal_heuristic.cpp), multiple worker threads share progress counters without locks:

```cpp
std::atomic<int32_t> next_start{0};
std::atomic<int32_t> starts_done{0};
// each thread does: int my_start = next_start.fetch_add(1);
```

---

## 14. enum class — Type-Safe Enumerations

`enum class` is like Python's `enum.Enum` — named constants that can't be accidentally confused with plain integers.

```python
# Python
from enum import Enum

class Status(Enum):
    Optimal = 0
    Feasible = 1
    Infeasible = 2
    TimeLimit = 3
    Error = 4
```

```cpp
// C++ — src/core/solution.h:18-25
enum class Status {
  Optimal,
  Feasible,
  Infeasible,
  Unbounded,
  TimeLimit,
  Error
};
```

Usage is the same pattern:

```python
# Python
if result.status == Status.Optimal:
    print("solved!")
```

```cpp
// C++
if (result.status == SolveResult::Status::Optimal) {
    // solved
}
```

See also [src/model/highs_bridge.h:25-31](src/model/highs_bridge.h#L25-L31) for `RCFixingStrategy`.

---

## 15. Structured Bindings — Tuple Unpacking

C++17 structured bindings are exactly Python's tuple unpacking:

```python
# Python
queue_entry = (node, label_idx)
u, li = queue_entry
```

```cpp
// C++ — src/preprocess/edge_elimination.h:59
auto [u, li] = queue[head++];
// u gets queue[head].node, li gets queue[head].label_idx
```

This also works for pairs and tuples:

```cpp
auto [graph, caps] = builder.build();  // unpacks std::tuple<digraph, vector<double>>
```

See [src/core/digraph.h:106](src/core/digraph.h#L106) where the builder returns both values.

---

## 16. Exceptions

C++ exceptions work like Python exceptions — `throw` is `raise`, `catch` is `except`.

```python
# Python
def build(num_nodes: int, profits: list[float]) -> None:
    if len(profits) != num_nodes:
        raise ValueError("profits size must equal num_nodes")
```

```cpp
// C++ — src/core/problem.cpp:13-14
if (profits.size() != static_cast<size_t>(num_nodes))
  throw std::invalid_argument("profits size must equal num_nodes");
```

Common exception types:
- `std::invalid_argument` — bad input (like Python's `ValueError`)
- `std::runtime_error` — runtime failure (like Python's `RuntimeError`)
- `std::out_of_range` — index out of bounds (like Python's `IndexError`)

In this codebase, exceptions are used only at system boundaries (validating inputs to `Problem::build`, file I/O errors in `io.cpp`). Internal code relies on invariants rather than defensive checks.

---

## 17. Standard Algorithms

The `<algorithm>` header provides generic algorithms that work on any container — the equivalent of Python's built-in functions.

```python
# Python
sorted(cuts, key=lambda c: c.violation, reverse=True)
all(x % 1 == 0 for x in values)
```

```cpp
// C++
std::sort(cuts.begin(), cuts.end(), [](const Cut& a, const Cut& b) {
  return a.violation > b.violation;
});
std::all_of(values.begin(), values.end(), [](double x) {
  return x == std::floor(x);
});
```

Common algorithms used in this codebase:

| C++ | Python equivalent | Where |
|-----|-------------------|-------|
| `std::sort` | `list.sort()` | [src/sep/cut_selector.cpp](src/sep/cut_selector.cpp) |
| `std::iota` | `range()` fill | [src/core/digraph.h:66](src/core/digraph.h#L66) |
| `std::all_of` | `all(...)` | [src/model/highs_bridge.cpp](src/model/highs_bridge.cpp) |
| `std::find` | `list.index()` | [src/model/model.cpp](src/model/model.cpp) |
| `std::erase_if` | list comprehension filter | [src/preprocess/edge_elimination.h](src/preprocess/edge_elimination.h) |
| `std::min` / `std::max` | `min()` / `max()` | everywhere |

The `.begin()` and `.end()` calls return iterators — think of them as the start and end of a Python range. Range-for loops hide this entirely:

```cpp
for (const auto& cut : cuts) { ... }  // equivalent to: for cut in cuts:
```

---

## 18. nanobind — C++ ↔ Python Bridge

nanobind is the library that makes the C++ solver callable from Python. It wraps C++ classes and functions so they appear as normal Python objects.

See [python/bindings.cpp](python/bindings.cpp).

### Module definition

```cpp
// python/bindings.cpp:27
NB_MODULE(_cptp, m) {
  m.doc() = "CPTP Branch-and-Cut Solver";
  // register everything below
}
```

This is like writing a `__init__.py` that exposes classes.

### Exposing a C++ enum to Python

```cpp
// python/bindings.cpp:31-37
nb::enum_<cptp::SolveResult::Status>(m, "Status")
  .value("Optimal", cptp::SolveResult::Status::Optimal)
  .value("Feasible", cptp::SolveResult::Status::Feasible)
  ...
```

After this, in Python: `from _cptp import Status; Status.Optimal`

### Exposing a C++ class

```cpp
// python/bindings.cpp:48-68
nb::class_<cptp::SolveResult>(m, "SolveResult")
  .def_ro("status", &cptp::SolveResult::status)    // read-only property
  .def_ro("objective", &cptp::SolveResult::objective)
  .def("is_optimal", &cptp::SolveResult::is_optimal)  // method
  .def_prop_ro("tour", [](cptp::SolveResult& self) {
    return vec_view_numpy<int32_t>(self.tour, nb::find(self));
  })  // property with custom getter (returns numpy array, zero-copy)
```

### Zero-copy numpy arrays

The helper at [python/bindings.cpp:20-25](python/bindings.cpp#L20-L25) wraps a C++ `std::vector<T>` as a numpy array without copying the data:

```cpp
template <typename T>
static nb::ndarray<nb::numpy, const T, nb::shape<-1>> vec_view_numpy(
    const std::vector<T>& v, nb::handle parent) {
  return nb::ndarray<...>(const_cast<T*>(v.data()), {v.size()}, parent);
}
```

The `parent` argument keeps the Python object alive so the underlying C++ vector isn't freed while numpy is still using its memory.

---

## 19. CMake — The Build System

CMake is like `pyproject.toml` + `setup.py`, but for compiled code. It describes how to compile the C++ source files into a library and executable.

Key parts of [CMakeLists.txt](CMakeLists.txt):

```cmake
# Require C++23
set(CMAKE_CXX_STANDARD 23)

# Build the solver as a static library
add_library(cptp STATIC
  src/core/problem.cpp
  src/sep/sec_separator.cpp
  ...
)

# Build the command-line tool
add_executable(cptp-solve src/cli/main.cpp)
target_link_libraries(cptp-solve PRIVATE cptp)

# Build Python bindings (optional)
nanobind_add_module(_cptp python/bindings.cpp)
target_link_libraries(_cptp PRIVATE cptp)
```

To build:

```bash
cmake -B build          # configure (like pip install --no-build-isolation)
cmake --build build -j  # compile (like python setup.py build_ext)
```

---

## 20. How the Solver Fits Together

Here is the data flow and which C++ concept powers each part:

```
Python / CLI
    │
    ▼
Model (model/model.h)           ← user-facing API, holds Problem + Logger
    │  set_problem(), solve()
    ▼
HiGHSBridge (model/highs_bridge.h)   ← wires everything into HiGHS
    │  build_formulation()
    │  install_separators()
    ▼
HiGHS (third-party MIP solver)
    │
    ├── cut callback ──────────► SeparationOracle::separate()
    │                                │
    │                                ├── SECSeparator::separate()   ─┐
    │                                ├── RCISeparator::separate()    │  Virtual functions
    │                                ├── CombSeparator::separate()   │  (polymorphism)
    │                                └── ...                        ─┘
    │
    ├── heuristic callback ────► PrimalHeuristic::run()
    │                             (parallel ILS, atomics + jthreads)
    │
    └── propagator callback ──► edge_elimination, labeling_from()
                                  (span views, const correctness)
    │
    ▼
SolveResult                     ← returned to caller
```

**Which concept powers what:**

| Layer | Key C++ concept |
|-------|-----------------|
| Separators are interchangeable | Virtual functions + `unique_ptr` |
| No copies of large data | `std::span`, move semantics |
| Thread-safe parallel heuristic | `std::atomic`, `std::jthread`, RAII locks |
| Callbacks capture solver state | Lambdas with `[&]` capture |
| Python can call C++ | nanobind bindings |
| Compile-time constants | `constexpr` |

---

## 21. Quick Reference Cheat Sheet

| Python | C++ | Notes |
|--------|-----|-------|
| `import foo` | `#include "foo.h"` | |
| `import foo.bar` | `namespace foo::bar` | declared in source, not imported |
| `list[int]` | `std::vector<int32_t>` | must specify element type |
| `dict[str, int]` | `std::map<std::string, int32_t>` | or `unordered_map` for hash map |
| `tuple[int, float]` | `std::pair<int, double>` or `std::tuple<...>` | |
| `a, b = tup` | `auto [a, b] = tup;` | structured bindings |
| `def f(x: int) -> str` | `std::string f(int32_t x)` | return type comes first |
| `lambda x: x + 1` | `[](int x) { return x + 1; }` | capture list `[]` |
| `lambda` with closure | `[&](int x) { return x + offset; }` | `&` captures outer vars |
| `ABC` + `@abstractmethod` | `virtual method() = 0;` | pure virtual |
| `class B(A)` | `class B : public A` | |
| `@override` (Python 3.12) | `override` keyword | compiler-checked |
| `Final[int]` | `const int32_t` or `constexpr` | |
| `Optional[T]` | `std::optional<T>` | |
| `raise ValueError(...)` | `throw std::invalid_argument(...)` | |
| `with lock:` | `std::lock_guard lock(mu_);` | RAII |
| `t.join()` | `std::jthread` auto-joins | |
| `threading.Event` | `std::atomic<bool>` | |
| `x = big_list; big_list = []` | `auto x = std::move(big_list);` | transfer without copy |
| `memoryview` | `std::span<const T>` | zero-copy view |
| GC frees objects | `unique_ptr` / `shared_ptr` | explicit ownership |
| `None` | `nullptr` (for pointers) | |
| `T` (generic type) | `template <typename T>` | compile-time generics |
| `sorted(x, key=f)` | `std::sort(x.begin(), x.end(), cmp)` | |
| `all(pred(x) for x in xs)` | `std::all_of(xs.begin(), xs.end(), pred)` | |
| `setup.py` / `pyproject.toml` | `CMakeLists.txt` | |
| `from enum import Enum` | `enum class` | |
