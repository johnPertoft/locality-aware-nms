package(default_visibility = ["//visibility:public"])

py_library(
    name = "lanms_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":lanms_ops_py",
    ],
    srcs_version = "PY3",
)

# TODO: Only run tests like this?
#py_test(
#    name = "lanms_ops_test_py",
#    srcs = [
#        "python/ops/nms_ops_test.py"
#    ],
#    main = "",
#    deps = [
#        ":lanms_ops_py
#    ],
#    srcs_version = "PY3",
#)

py_library(
    name = "lanms_ops_py",
    srcs = ([
        "python/ops/nms_ops.py",
    ]),
    data = [
        ":python/ops/_nms_ops.so"
    ],
    srcs_version = "PY3",
)

cc_binary(
    name = "python/ops/_nms_ops.so",
    srcs = [
        "cc/kernels/geom.cc",
        "cc/kernels/geom.h",
        "cc/kernels/nms.cc",
        "cc/kernels/nms.h",
        "cc/kernels/nms_kernels.cc",
        "cc/ops/nms_ops.cc",
    ],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    linkshared = 1,
    copts = [
        "-pthread",
        "-std=c++11",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
)

cc_test(
    name = "nms_test",
    srcs = [
        "cc/kernels/geom.cc",
        "cc/kernels/geom.h",
        "cc/kernels/geom_test.h",
        "cc/kernels/nms.cc",
        "cc/kernels/nms.h",
        "cc/kernels/nms_test.h",
        "cc/kernels/tests_main.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
)
