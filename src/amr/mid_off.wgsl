// Per-block timesteps are off: no block is ever part way through a step.
fn mid_block(slot: i32) -> bool { return false; }
fn mid_cons(slot: i32, l: vec3<i32>) -> Cons { return u_leaf(slot, l); }
