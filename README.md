# matreex

[![Crates.io](https://img.shields.io/crates/v/matreex.svg)](https://crates.io/crates/matreex)
[![Documentation](https://docs.rs/matreex/badge.svg)](https://docs.rs/matreex)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A simple matrix implementation.

## Quick Start

```rust
let lhs = matrix![[0, 1, 2], [3, 4, 5]];
let rhs = matrix![[2, 2, 2], [2, 2, 2]];
assert_eq!(lhs + rhs, matrix![[2, 3, 4], [5, 6, 7]]);

let lhs = matrix![[0, 1, 2], [3, 4, 5]];
let rhs = matrix![[0, 1], [2, 3], [4, 5]];
assert_eq!(lhs * rhs, matrix![[10, 13], [28, 40]]);

let matrix = matrix![[0, 1, 2], [3, 4, 5]];
assert_eq!(matrix - 2, matrix![[-2, -1, 0], [1, 2, 3]]);

let matrix = matrix![[0, 1, 2], [3, 4, 5]];
assert_eq!(2 - matrix, matrix![[2, 1, 0], [-1, -2, -3]]);
```

## FAQs

### Why named `matreex`?

Hmm ... Who knows? Could be a name conflict.
