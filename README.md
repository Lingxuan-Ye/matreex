# matreex

[![Crates.io](https://img.shields.io/crates/v/matreex.svg)](https://crates.io/crates/matreex)
[![Documentation](https://docs.rs/matreex/badge.svg)](https://docs.rs/matreex)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A simple matrix implementation.

## Quick Start

```rust
use matreex::matrix;

let lhs = matrix![[0, 1, 2], [3, 4, 5]];
let rhs = matrix![[0, 1], [2, 3], [4, 5]];
assert_eq!(lhs * rhs, matrix![[10, 13], [28, 40]]);
```

## FAQs

### Why named `matreex`?

Hmm ... Who knows? Could be a name conflict.
