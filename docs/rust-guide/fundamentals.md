# Rust Fundamentals for RustyStats

This guide teaches the Rust concepts you need to understand and maintain the RustyStats codebase. It's written for programmers who know another language (Python, C++, Java, etc.) but are new to Rust.

**What makes Rust different**: Rust guarantees memory safety at compile time without garbage collection. This is achieved through its *ownership system*, which may feel unfamiliar at first but becomes natural with practice.

---

## Part 1: Ownership and Borrowing

This is Rust's most distinctive feature. Understanding it is essential for reading and writing Rust code.

### 1.1 The Problem Rust Solves

In C/C++, memory bugs are common:
- **Use after free**: accessing memory after it's been deallocated
- **Double free**: deallocating the same memory twice
- **Dangling pointers**: pointers to deallocated memory
- **Data races**: multiple threads accessing memory unsafely

Languages like Python and Java solve this with garbage collection (GC), but GC has overhead and unpredictable pauses.

Rust solves it at compile time with zero runtime cost through **ownership rules**.

### 1.2 The Three Rules of Ownership

1. **Each value has exactly one owner** (a variable that "owns" it)
2. **When the owner goes out of scope, the value is dropped** (memory is freed)
3. **Ownership can be transferred (moved) or temporarily lent (borrowed)**

### 1.3 Ownership and Moves

```rust
fn main() {
    let s1 = String::from("hello");  // s1 owns the string
    let s2 = s1;                      // Ownership MOVES to s2
    
    // println!("{}", s1);  // ERROR! s1 no longer owns anything
    println!("{}", s2);     // OK: s2 owns the string
}
```

When `s2 = s1`, the ownership **moves**. The variable `s1` becomes invalid—Rust prevents you from using it.

!!! info "Why Move Instead of Copy?"
    `String` stores its data on the heap. If Rust copied the data, both `s1` and `s2` would point to the same memory. When both go out of scope, we'd have a double-free bug. Instead, Rust transfers ownership.

**Simple types are copied, not moved:**

```rust
let x = 5;      // i32 is a simple type stored on the stack
let y = x;      // x is COPIED to y (no move)
println!("{} {}", x, y);  // Both are valid: "5 5"
```

Types like integers, floats, and booleans implement the `Copy` trait and are copied automatically.

### 1.4 Borrowing with References

What if you want to use a value without taking ownership? You **borrow** it with a reference:

```rust
fn main() {
    let s = String::from("hello");
    
    let len = calculate_length(&s);  // Pass a reference (borrow)
    
    println!("Length of '{}' is {}", s, len);  // s is still valid!
}

fn calculate_length(s: &String) -> usize {  // s is a reference
    s.len()
}  // s goes out of scope, but since it doesn't own the String, nothing is dropped
```

The `&` creates a reference. The function borrows `s` temporarily but doesn't own it.

### 1.5 Mutable References

By default, references are immutable. To modify borrowed data:

```rust
fn main() {
    let mut s = String::from("hello");  // s must be declared mut
    
    change(&mut s);  // Pass a mutable reference
    
    println!("{}", s);  // "hello, world"
}

fn change(s: &mut String) {
    s.push_str(", world");
}
```

**Critical rule**: You can have either:
- **One mutable reference**, OR
- **Any number of immutable references**

But not both at the same time! This prevents data races at compile time.

```rust
let mut s = String::from("hello");

let r1 = &s;      // OK: immutable borrow
let r2 = &s;      // OK: another immutable borrow
// let r3 = &mut s;  // ERROR! Can't borrow mutably while immutably borrowed

println!("{} {}", r1, r2);

let r3 = &mut s;  // OK: r1 and r2 are no longer used
```

### 1.6 Ownership in RustyStats

In RustyStats, you'll see patterns like:

```rust
// Taking ownership (consumes the input)
pub fn process(data: Array1<f64>) -> Array1<f64> {
    // data is owned here, will be dropped when function returns
    data * 2.0
}

// Borrowing (doesn't consume)
pub fn calculate_mean(data: &Array1<f64>) -> f64 {
    data.sum() / data.len() as f64
}

// Mutable borrowing (modifies in place)
pub fn normalize(data: &mut Array1<f64>) {
    let mean = data.mean().unwrap();
    *data -= mean;
}
```

**Guideline**: Prefer borrowing (`&` or `&mut`) unless you need ownership.

---

## Part 2: Structs and Methods

### 2.1 Defining Structs

Structs group related data:

```rust
pub struct IRLSConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub min_weight: f64,
    pub verbose: bool,
}
```

- `pub` makes the struct and its fields accessible from other modules
- Each field has a name and type

### 2.2 Creating Instances

```rust
let config = IRLSConfig {
    max_iterations: 25,
    tolerance: 1e-8,
    min_weight: 1e-10,
    verbose: false,
};

// Field shorthand when variable name matches
let max_iterations = 50;
let config2 = IRLSConfig {
    max_iterations,  // Same as max_iterations: max_iterations
    ..config         // Copy remaining fields from config
};
```

### 2.3 Methods with impl Blocks

Methods are defined in `impl` blocks:

```rust
impl IRLSConfig {
    // Associated function (no self) - like a static method
    pub fn new() -> Self {
        Self {
            max_iterations: 25,
            tolerance: 1e-8,
            min_weight: 1e-10,
            verbose: false,
        }
    }
    
    // Method (takes self) - called on an instance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self  // Return self for chaining
    }
    
    // Method that borrows self (doesn't consume)
    pub fn is_converged(&self, old_dev: f64, new_dev: f64) -> bool {
        (old_dev - new_dev).abs() / old_dev < self.tolerance
    }
    
    // Method that mutably borrows self
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }
}

// Usage
let config = IRLSConfig::new()
    .with_tolerance(1e-6);  // Builder pattern
```

### 2.4 The self Parameter

- `self` — takes ownership, method consumes the instance
- `&self` — immutable borrow, method can read but not modify
- `&mut self` — mutable borrow, method can modify

Most methods use `&self` or `&mut self`.

---

## Part 3: Traits (Interfaces)

Traits define shared behavior, similar to interfaces in other languages.

### 3.1 Defining Traits

```rust
pub trait Family {
    /// Name of the distribution family
    fn name(&self) -> &str;
    
    /// Variance function V(μ)
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64>;
    
    /// Unit deviance d(y, μ)
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64>;
    
    /// Default link function for this family
    fn default_link(&self) -> Box<dyn Link>;
    
    /// Initialize μ from y
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64>;
    
    /// Check if μ values are valid
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool;
}
```

### 3.2 Implementing Traits

```rust
pub struct PoissonFamily;

impl Family for PoissonFamily {
    fn name(&self) -> &str {
        "Poisson"
    }
    
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.clone()  // V(μ) = μ for Poisson
    }
    
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        // d(y, μ) = 2[y log(y/μ) - (y - μ)]
        Zip::from(y).and(mu).map_collect(|&yi, &mui| {
            if yi > 0.0 {
                2.0 * (yi * (yi / mui).ln() - (yi - mui))
            } else {
                2.0 * mui
            }
        })
    }
    
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogLink)
    }
    
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        y.mapv(|yi| (yi + 0.1).max(0.1))
    }
    
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
        mu.iter().all(|&m| m > 0.0)
    }
}
```

### 3.3 Trait Objects and Dynamic Dispatch

When the concrete type isn't known at compile time, use trait objects:

```rust
// This function works with ANY type implementing Family
pub fn fit_glm(
    y: &Array1<f64>,
    x: &Array2<f64>,
    family: &dyn Family,  // Trait object: any Family implementation
    link: &dyn Link,
) -> Result<IRLSResult> {
    let mu = family.initialize_mu(y);
    let variance = family.variance(&mu);
    // ...
}

// Calling with different families
let result1 = fit_glm(&y, &x, &PoissonFamily, &LogLink)?;
let result2 = fit_glm(&y, &x, &GaussianFamily, &IdentityLink)?;
```

The `dyn` keyword indicates dynamic dispatch—the method to call is determined at runtime.

### 3.4 Trait Bounds

Require types to implement traits:

```rust
// T must implement both Send and Sync (safe for parallel use)
pub trait Family: Send + Sync {
    // ...
}

// Generic function with trait bound
fn process<T: Family>(family: &T) {
    println!("Family: {}", family.name());
}

// Alternative syntax with where clause
fn process<T>(family: &T) 
where 
    T: Family + Clone,
{
    // ...
}
```

### 3.5 Common Standard Library Traits

| Trait | Purpose | Example |
|-------|---------|---------|
| `Clone` | Deep copy | `let b = a.clone()` |
| `Copy` | Implicit bitwise copy | Integers, floats |
| `Debug` | Debug formatting | `println!("{:?}", x)` |
| `Display` | User-facing formatting | `println!("{}", x)` |
| `Default` | Default value | `let x = T::default()` |
| `Send` | Safe to send between threads | Required for Rayon |
| `Sync` | Safe to share references between threads | Required for Rayon |

---

## Part 4: Error Handling

Rust doesn't have exceptions. Instead, it uses the `Result` type for recoverable errors.

### 4.1 The Result Type

```rust
enum Result<T, E> {
    Ok(T),   // Success, contains a value of type T
    Err(E),  // Error, contains an error of type E
}
```

Functions that can fail return `Result`:

```rust
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err(String::from("Division by zero"))
    } else {
        Ok(a / b)
    }
}
```

### 4.2 Handling Results

**Option 1: Pattern matching**

```rust
match divide(10.0, 2.0) {
    Ok(result) => println!("Result: {}", result),
    Err(e) => println!("Error: {}", e),
}
```

**Option 2: if let**

```rust
if let Ok(result) = divide(10.0, 2.0) {
    println!("Result: {}", result);
}
```

**Option 3: The ? operator (propagate errors)**

```rust
fn calculate() -> Result<f64, String> {
    let a = divide(10.0, 2.0)?;  // Returns early if Err
    let b = divide(a, 3.0)?;
    Ok(b)
}
```

The `?` operator:
1. If `Result` is `Ok`, unwrap the value and continue
2. If `Result` is `Err`, return the error immediately

### 4.3 Custom Error Types

RustyStats defines its own error type:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RustyStatsError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },
    
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Linear algebra error: {0}")]
    LinAlgError(String),
}

// Type alias for convenience
pub type Result<T> = std::result::Result<T, RustyStatsError>;
```

Usage:

```rust
pub fn fit_glm(y: &Array1<f64>, x: &Array2<f64>) -> Result<IRLSResult> {
    if y.len() != x.nrows() {
        return Err(RustyStatsError::DimensionMismatch {
            expected: x.nrows(),
            actual: y.len(),
        });
    }
    // ...
}
```

### 4.4 The Option Type

For values that might be absent (not an error, just missing):

```rust
enum Option<T> {
    Some(T),  // Value present
    None,     // Value absent
}

fn find_max(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        None
    } else {
        Some(data.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    }
}

// Usage
match find_max(&data) {
    Some(max) => println!("Max: {}", max),
    None => println!("Empty data"),
}

// Or with unwrap_or
let max = find_max(&data).unwrap_or(0.0);
```

---

## Part 5: Iterators

Iterators are Rust's powerful abstraction for processing sequences.

### 5.1 Basic Iteration

```rust
let numbers = vec![1, 2, 3, 4, 5];

// Iterate by reference (most common)
for n in &numbers {
    println!("{}", n);
}

// Iterate by mutable reference
let mut numbers = vec![1, 2, 3, 4, 5];
for n in &mut numbers {
    *n *= 2;  // Double each element
}

// Iterate by value (consumes the vector)
for n in numbers {
    println!("{}", n);
}
// numbers is no longer usable here
```

### 5.2 Iterator Methods

Iterators have powerful combinators:

```rust
let numbers = vec![1, 2, 3, 4, 5];

// map: transform each element
let squared: Vec<_> = numbers.iter().map(|x| x * x).collect();
// [1, 4, 9, 16, 25]

// filter: keep elements matching a condition
let evens: Vec<_> = numbers.iter().filter(|&x| x % 2 == 0).collect();
// [2, 4]

// fold: accumulate into a single value
let sum: i32 = numbers.iter().fold(0, |acc, x| acc + x);
// 15

// chain multiple operations
let result: Vec<_> = numbers.iter()
    .filter(|&x| x % 2 == 1)  // Keep odd numbers
    .map(|x| x * 2)            // Double them
    .collect();
// [2, 6, 10]
```

### 5.3 ndarray Iterators

RustyStats uses ndarray, which has its own iteration patterns:

```rust
use ndarray::{Array1, Array2, Zip};

let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

// Element-wise iteration with Zip
let c = Zip::from(&a).and(&b).map_collect(|&ai, &bi| ai + bi);
// [5.0, 7.0, 9.0]

// Iterate over rows of a 2D array
let matrix = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
for row in matrix.rows() {
    println!("{:?}", row);
}

// mapv: map over values (creates new array)
let doubled = a.mapv(|x| x * 2.0);

// mapv_inplace: modify in place
let mut a = a;
a.mapv_inplace(|x| x * 2.0);
```

### 5.4 Parallel Iterators with Rayon

Rayon makes parallelism easy:

```rust
use rayon::prelude::*;

let numbers: Vec<i32> = (0..1000000).collect();

// Sequential
let sum: i32 = numbers.iter().sum();

// Parallel (just change iter() to par_iter())
let sum: i32 = numbers.par_iter().sum();

// Parallel map
let squared: Vec<_> = numbers.par_iter().map(|x| x * x).collect();

// Parallel fold-reduce
let sum = (0..1000000).into_par_iter()
    .fold(|| 0, |acc, x| acc + x)  // Fold per thread
    .reduce(|| 0, |a, b| a + b);   // Combine threads
```

RustyStats uses parallel iterators for expensive computations like X'WX.

---

## Part 6: Generics

Write code that works with multiple types.

### 6.1 Generic Functions

```rust
// Works with any type T that implements PartialOrd
fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

let x = max(5, 10);        // T = i32
let y = max(3.14, 2.71);   // T = f64
```

### 6.2 Generic Structs

```rust
struct Point<T> {
    x: T,
    y: T,
}

impl<T> Point<T> {
    fn new(x: T, y: T) -> Self {
        Point { x, y }
    }
}

// Can add methods only for specific types
impl Point<f64> {
    fn distance_from_origin(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}

let int_point = Point::new(5, 10);
let float_point = Point::new(1.0, 2.0);
let dist = float_point.distance_from_origin();
```

### 6.3 Monomorphization

Rust compiles generic code by generating specialized versions for each type used. This means:
- No runtime overhead (unlike Java generics)
- Larger binary size
- Fast execution

---

## Part 7: Modules and Visibility

### 7.1 Module Structure

```
crate_name/
├── src/
│   ├── lib.rs           // Crate root
│   ├── families/        // Module directory
│   │   ├── mod.rs       // Module root
│   │   ├── poisson.rs   // Submodule
│   │   └── gaussian.rs  // Submodule
│   └── solvers/
│       ├── mod.rs
│       └── irls.rs
```

In `lib.rs`:
```rust
pub mod families;  // Declares the families module
pub mod solvers;
```

In `families/mod.rs`:
```rust
mod poisson;   // Private submodule
mod gaussian;

pub use poisson::PoissonFamily;   // Re-export publicly
pub use gaussian::GaussianFamily;

pub trait Family { ... }
```

### 7.2 Visibility

- `pub` — public, accessible from anywhere
- `pub(crate)` — public within the crate only
- `pub(super)` — public to parent module only
- (nothing) — private to current module

```rust
pub struct Config {
    pub max_iter: usize,       // Public field
    pub(crate) internal: bool, // Crate-only field
    private: String,           // Private field
}
```

### 7.3 Use Statements

```rust
// Import specific items
use crate::families::{Family, PoissonFamily};

// Import everything from a module
use crate::families::*;

// Rename on import
use std::collections::HashMap as Map;

// Re-export
pub use crate::families::Family;
```

---

## Part 8: Common Patterns in RustyStats

### 8.1 Builder Pattern

```rust
let config = IRLSConfig::new()
    .max_iterations(50)
    .tolerance(1e-6)
    .verbose(true);
```

Implementation:
```rust
impl IRLSConfig {
    pub fn new() -> Self { ... }
    
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }
    
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
}
```

### 8.2 Trait Objects for Polymorphism

```rust
fn fit_glm(family: &dyn Family, link: &dyn Link) { ... }

// Works with any Family implementation
fit_glm(&PoissonFamily, &LogLink);
fit_glm(&GaussianFamily, &IdentityLink);
```

### 8.3 Result Propagation

```rust
pub fn fit() -> Result<IRLSResult> {
    let mu = initialize()?;      // ? propagates errors
    let weights = compute_weights(&mu)?;
    let beta = solve(&weights)?;
    Ok(IRLSResult { ... })
}
```

### 8.4 Parallel Computation

```rust
use rayon::prelude::*;

let result = (0..n).into_par_iter()
    .fold(|| init_value, |acc, i| accumulate(acc, i))
    .reduce(|| init_value, |a, b| combine(a, b));
```

---

## Summary

Key Rust concepts for RustyStats:

1. **Ownership**: Values have one owner; use `&` to borrow
2. **Structs**: Group data; add methods with `impl`
3. **Traits**: Define interfaces; use `dyn Trait` for polymorphism
4. **Result/Option**: Handle errors and missing values explicitly
5. **Iterators**: Powerful, composable, parallelizable with Rayon
6. **Generics**: Write type-safe reusable code
7. **Modules**: Organize code; control visibility with `pub`

---

## Further Reading

- [The Rust Book](https://doc.rust-lang.org/book/) — The official guide
- [Rust By Example](https://doc.rust-lang.org/rust-by-example/) — Learn through examples
- [ndarray documentation](https://docs.rs/ndarray/) — Array library used by RustyStats
- [Rayon documentation](https://docs.rs/rayon/) — Parallel iteration
