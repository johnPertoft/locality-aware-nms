LIB_SO := python/ops/_nms_ops.so

HEADER_FILES := $(shell find . -name "*.h" ! -name "*test*")
CPP_FILES := $(shell find . -name "*.cc" ! -name "*test*")

TF_LFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
TF_CFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')

CFLAGS = $(TF_CFLAGS) -fPIC -O2 -std=c++11
LDFLAGS = -shared $(TF_LFLAGS)

COMPILER := g++

.PHONY: clean

$(LIB_SO): $(HEADER_FILES) $(CPP_FILES)
		$(COMPILER) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
		rm -f $(LIB_SO)
