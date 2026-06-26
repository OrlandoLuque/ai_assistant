macro_rules! assert_eq_test {
    ($left:expr, $right:expr) => {
        if $left != $right {
            return Err(format!("expected {:?}, got {:?}", $right, $left));
        }
    };
    ($left:expr, $right:expr, $msg:expr) => {
        if $left != $right {
            return Err(format!("{}: expected {:?}, got {:?}", $msg, $right, $left));
        }
    };
}

macro_rules! assert_test {
    ($cond:expr) => {
        if !$cond {
            return Err(format!("assertion failed: {}", stringify!($cond)));
        }
    };
    ($cond:expr, $msg:expr) => {
        if !$cond {
            return Err(format!("{}", $msg));
        }
    };
}
