export module alp;

export import :set;
export import :map;

// Export backend interface partitions
export import :backend_sse;

#if defined(ALP_USE_EVE)
export import :backend_eve;
#endif

#if defined(ALP_USE_EXPERIMENTAL_SIMD)
export import :backend_std_simd;
#endif