package(default_visibility = ["//visibility:public"])

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
    copts = [
        "-pthread",
        "-std=c++11",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-fPIC",
    ],
    deps = [
        "@com_google_googletest//:gtest",
    ],
)

#cc_binary(
#    name = "python/ops/_nms_ops.so",
#    srcs = [
#        "cc/kernels/geom.h",
#        "cc/kernels/nms.h",
#        "cc/kernels/nms_kernels.cc",
#        "cc/ops/nms_ops.cc",
#    ],
#    copts = [
#        "-pthread",
#        "-std=c++11",
#        "-D_GLIBCXX_USE_CXX11_ABI=0",
#        "-fPIC",
#    ],
#    linkshared = 1,
#    deps = [
#        "@local_config_tf//:libtensorflow_framework",
#        "@local_config_tf//:tf_header_lib",
#    ],
#)
#
#py_library(
#    name = "nms_ops_py",
#    srcs = ([
#        "python/ops/nms_ops.py",
#    ]),
#    data = [
#        ":python/ops/_nms_ops.so",
#    ],
#    srcs_version = "PY2AND3",
#)
#
#py_library(
#    name = "nms_py",
#    srcs = ([
#        "__init__.py",
#        "python/__init__.py",
#        "python/ops/__init__.py",
#    ]),
#    srcs_version = "PY2AND3",
#    deps = [
#        ":nms_ops_py",
#    ],
#)

cc_library(
    name = "nms_ops",
    srcs = [
        "cc/kernels/geom.cc",
        "cc/kernels/nms.cc",
        "cc/kernels/nms_kernels.cc",
        "cc/ops/nms_ops.cc",
    ],
    hdrs = [
        "cc/kernels/geom.h",
        "cc/kernels/nms.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)
