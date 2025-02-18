# matreex

[![Crates.io](https://img.shields.io/crates/v/matreex.svg)](https://crates.io/crates/matreex)
[![Documentation](https://docs.rs/matreex/badge.svg)](https://docs.rs/matreex)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A simple matrix implementation.

## Quick Start

First, we need to import `matrix!`.

```rust
use matreex::matrix;
```

### Addition

```rust
let lhs = matrix![[1, 2, 3], [4, 5, 6]];
let rhs = matrix![[2, 2, 2], [2, 2, 2]];
assert_eq!(lhs + rhs, matrix![[3, 4, 5], [6, 7, 8]]);
```

### Subtraction

```rust
let lhs = matrix![[1, 2, 3], [4, 5, 6]];
let rhs = matrix![[2, 2, 2], [2, 2, 2]];
assert_eq!(lhs - rhs, matrix![[-1, 0, 1], [2, 3, 4]]);
```

### Multiplication

```rust
let lhs = matrix![[1, 2, 3], [4, 5, 6]];
let rhs = matrix![[1, 2], [3, 4], [5, 6]];
assert_eq!(lhs * rhs, matrix![[22, 28], [49, 64]]);
```

### Division

```rust
let lhs = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
let rhs = matrix![[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]];
assert_eq!(lhs / rhs, matrix![[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]);
```

Wait, matrix division isn't well-defined, remember? It won't compile. But
don't worry, you might just need to perform elementwise division:

```rust
let lhs = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
let rhs = matrix![[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]];
assert_eq!(lhs.elementwise_div(&rhs), Ok(matrix![[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]));
```

Or scalar division:

```rust
let matrix = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
assert_eq!(matrix / 2.0, matrix![[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]);

let matrix = matrix![[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]];
assert_eq!(2.0 / matrix, matrix![[2.0, 1.0, 0.5], [0.25, 0.125, 0.0625]]);
```

Or maybe the inverse of a matrix?

Nah, we don't have that yet.

## FAQs

### Why named `matreex`?

Hmm ... Who knows? Could be a name conflict.
