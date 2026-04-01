macro_rules! uninit {
    ($length:expr) => {{
        let mut vec = Vec::with_capacity($length);
        #[allow(clippy::uninit_vec)]
        unsafe {
            vec.set_len($length);
        }
        vec
    }};
    () => {{
        #[allow(clippy::uninit_assumed_init, invalid_value)]
        unsafe {
            std::mem::MaybeUninit::uninit().assume_init()
        }
    }};
}

pub(crate) use uninit;
