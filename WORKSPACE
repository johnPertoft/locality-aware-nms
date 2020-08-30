load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("//tf:tf_configure.bzl", "tf_configure")

git_repository(
    name = "googletest",
    remote = "https://github.com/google/googletest",
    branch = "release-1.8.1",
)

tf_configure(name = "local_config_tf")
