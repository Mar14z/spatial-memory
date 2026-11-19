// Print input/output tensor specs of an OM model.
#include <acl/acl.h>

#include <cstdio>

static const char* dtype_name(aclDataType t) {
    switch (t) {
        case ACL_FLOAT: return "fp32";
        case ACL_FLOAT16: return "fp16";
        case ACL_INT8: return "int8";
        case ACL_UINT8: return "uint8";
        case ACL_INT32: return "int32";
        default: return "other";
    }
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "/root/clip_vision_om.om";
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtContext ctx = nullptr;
    aclrtCreateContext(&ctx, 0);

    uint32_t mid = 0;
    if (aclmdlLoadFromFile(path, &mid) != ACL_SUCCESS) {
        printf("load failed: %s\n", path);
        return 1;
    }
    aclmdlDesc* d = aclmdlCreateDesc();
    aclmdlGetDesc(d, mid);
    size_t nin = aclmdlGetNumInputs(d), nout = aclmdlGetNumOutputs(d);
    printf("model=%s inputs=%zu outputs=%zu\n", path, nin, nout);

    for (size_t i = 0; i < nin; i++) {
        aclmdlIODims dims{};
        aclmdlGetInputDims(d, i, &dims);
        printf("IN[%zu] dtype=%s bytes=%zu dims=", i,
               dtype_name(aclmdlGetInputDataType(d, i)),
               aclmdlGetInputSizeByIndex(d, i));
        for (uint32_t k = 0; k < dims.dimCount; k++) printf("%lld ", (long long)dims.dims[k]);
        printf("\n");
    }
    for (size_t i = 0; i < nout; i++) {
        aclmdlIODims dims{};
        aclmdlGetOutputDims(d, i, &dims);
        printf("OUT[%zu] dtype=%s bytes=%zu dims=", i,
               dtype_name(aclmdlGetOutputDataType(d, i)),
               aclmdlGetOutputSizeByIndex(d, i));
        for (uint32_t k = 0; k < dims.dimCount; k++) printf("%lld ", (long long)dims.dims[k]);
        printf("\n");
    }

    aclmdlDestroyDesc(d);
    aclmdlUnload(mid);
    aclrtDestroyContext(ctx);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
