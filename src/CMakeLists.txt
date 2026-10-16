add_library(librw
    "${PROJECT_SOURCE_DIR}/args.h"
    "${PROJECT_SOURCE_DIR}/rw.h"

    anim.cpp
    base.cpp
    bmp.cpp
    camera.cpp
    charset.cpp
    clump.cpp
    engine.cpp
    error.cpp
    frame.cpp
    geometry.cpp
    geoplg.cpp
    hanim.cpp
    image.cpp
    light.cpp
    matfx.cpp
    pipeline.cpp
    plg.cpp
    png.cpp
    prim.cpp
    raster.cpp
    render.cpp
    rwanim.h
    rwengine.h
    rwerror.h
    rwobjects.h
    rwpipeline.h
    rwplg.h
    rwplugins.h
    rwrender.h
    rwuserdata.h
    skin.cpp
    texture.cpp
    tga.cpp
    tristrip.cpp
    userdata.cpp
    uvanim.cpp
    world.cpp

    d3d/d3d8.cpp
    d3d/d3d8matfx.cpp
    d3d/d3d8render.cpp
    d3d/d3d8skin.cpp
    d3d/d3d9.cpp
    d3d/d3d9matfx.cpp
    d3d/d3d9render.cpp
    d3d/d3d9skin.cpp
    d3d/d3d.cpp
    d3d/d3ddevice.cpp
    d3d/d3dimmed.cpp
    d3d/d3drender.cpp
    d3d/rwd3d8.h
    d3d/rwd3d9.h
    d3d/rwd3d.h
    d3d/rwd3dimpl.h
    d3d/rwxbox.h
    d3d/rwxboximpl.h
    d3d/xbox.cpp
    d3d/xboxmatfx.cpp
    d3d/xboxskin.cpp
    d3d/xboxvfmt.cpp

    gl/gl3.cpp
    gl/gl3device.cpp
    gl/gl3immed.cpp
    gl/gl3matfx.cpp
    gl/gl3pipe.cpp
    gl/gl3raster.cpp
    gl/gl3render.cpp
    gl/gl3shader.cpp
    gl/gl3skin.cpp
    gl/rwgl3.h
    gl/rwgl3impl.h
    gl/rwgl3plg.h
    gl/rwgl3shader.h
    gl/rwwdgl.h
    gl/wdgl.cpp
    gl/glad/glad.h
    gl/glad/glad.c
    gl/glad/khrplatform.h

    lodepng/lodepng.h
    lodepng/lodepng.cpp

    ps2/pds.cpp
    ps2/ps2.cpp
    ps2/ps2device.cpp
    ps2/ps2matfx.cpp
    ps2/ps2raster.cpp
    ps2/ps2skin.cpp
    ps2/rwps2.h
    ps2/rwps2impl.h
    ps2/rwps2plg.h
)
add_library(librw::librw ALIAS librw)

target_include_directories(librw
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${LIBRW_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(librw
    PRIVATE
        LODEPNG_NO_COMPILE_CPP
        "$<IF:$<CONFIG:DEBUG>,DEBUG,NDEBUG>"
    PUBLIC
        "RW_${LIBRW_PLATFORM}"
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_link_libraries(librw
        PRIVATE
            m
    )
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(librw
        PRIVATE
            "-Wall"
    )
    if (NOT LIBRW_PLATFORM_PS2)
        target_compile_options(librw
            PRIVATE
                "-Wextra"
                "-Wdouble-promotion"
                "-Wpedantic"
        )
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(librw
        PUBLIC
            /wd4996 /wd4244
    )
endif()

set_target_properties(librw
    PROPERTIES
        C_STANDARD 11
        C_EXTENSIONS OFF
        C_STANDARD_REQUIRED ON
        CXX_STANDARD 11
        CXX_EXTENSIONS OFF
        CXX_STANDARD_REQUIRED ON
        PREFIX ""
)

option(LIBRW_METALGL "Run the OpenGL backend on Metal (macOS)" OFF)
if(LIBRW_PLATFORM_GL3 AND LIBRW_METALGL)
    find_library(MTLGL_GLSLANG glslang PATHS /opt/homebrew/lib REQUIRED)
    find_library(MTLGL_GLSLANG_RES glslang-default-resource-limits PATHS /opt/homebrew/lib REQUIRED)
    find_library(MTLGL_SPIRV SPIRV PATHS /opt/homebrew/lib REQUIRED)
    find_library(MTLGL_SCROSS_MSL spirv-cross-msl PATHS /opt/homebrew/lib REQUIRED)
    find_library(MTLGL_SCROSS_GLSL spirv-cross-glsl PATHS /opt/homebrew/lib REQUIRED)
    find_library(MTLGL_SCROSS_CORE spirv-cross-core PATHS /opt/homebrew/lib REQUIRED)
    target_sources(librw PRIVATE mtl/mtlgl_impl.cpp mtl/mtlshadertrans.cpp mtl/mtlgl.h mtl/mtlshadertrans.h)
    set_source_files_properties(mtl/mtlgl_impl.cpp PROPERTIES COMPILE_OPTIONS "-x;objective-c++;-fobjc-arc;-std=c++17;-Wno-deprecated-declarations")
    set_source_files_properties(mtl/mtlshadertrans.cpp PROPERTIES COMPILE_OPTIONS "-std=c++17")
    target_include_directories(librw PRIVATE /opt/homebrew/include /opt/homebrew/include/glslang)
    target_compile_definitions(librw PUBLIC RW_METALGL)
    target_link_libraries(librw PUBLIC ${MTLGL_GLSLANG} ${MTLGL_GLSLANG_RES} ${MTLGL_SPIRV} ${MTLGL_SCROSS_MSL} ${MTLGL_SCROSS_GLSL}
        ${MTLGL_SCROSS_CORE} "-framework Metal" "-framework QuartzCore" "-framework Cocoa" "-framework Foundation")
endif()

if(LIBRW_PLATFORM_GL3)
    if (LIBRW_GL3_GFXLIB STREQUAL "GLFW")
        if (NOT TARGET glfw)
            find_package(glfw3 REQUIRED)
        endif()
        target_compile_definitions(librw PUBLIC LIBRW_GLFW)
        target_link_libraries(librw
            PUBLIC
                glfw
            )
    elseif (LIBRW_GL3_GFXLIB STREQUAL "SDL2")
        if (NOT TARGET SDL2::SDL2)
            find_package(SDL2 CONFIG)
            if (NOT TARGET SDL2::SDL2)
                find_package(SDL2 MODULE REQUIRED)
            endif()
        endif()
        target_compile_definitions(librw PUBLIC LIBRW_SDL2)
        target_link_libraries(librw
            PUBLIC
                SDL2::SDL2
        )
    elseif (LIBRW_GL3_GFXLIB STREQUAL "SDL3")
        find_package(SDL3 CONFIG REQUIRED)
        target_compile_definitions(librw PUBLIC LIBRW_SDL3)
        target_link_libraries(librw
            PUBLIC
                SDL3::SDL3
        )
    endif()

    set(OpenGL_GL_PREFERENCE GLVND)
    find_package(OpenGL)
    if(TARGET OpenGL::OpenGL)
        target_link_libraries(librw
            PRIVATE
                OpenGL::OpenGL
        )
    elseif(TARGET OpenGL::EGL)
        target_link_libraries(librw
            PRIVATE
                OpenGL::EGL
        )
    elseif(TARGET OpenGL::GL)
        target_link_libraries(librw
            PRIVATE
                OpenGL::GL
        )
    else()
        message(FATAL_ERROR "find_package(OpenGL) failed.")
    endif()
elseif(LIBRW_PLATFORM_D3D9)
    target_link_libraries(librw
        PRIVATE
            d3d9
            xinput
    )
endif()

if(LIBRW_INSTALL)
    install(
        FILES
            "${PROJECT_SOURCE_DIR}/args.h"
            "${PROJECT_SOURCE_DIR}/rw.h"
        DESTINATION "${LIBRW_INSTALL_INCLUDEDIR}"
    )
    install(
        FILES
            base.err
            rwbase.h
            rwcharset.h
            rwerror.h
            rwplg.h
            rwrender.h
            rwengine.h
            rwpipeline.h
            rwobjects.h
            rwanim.h
            rwplugins.h
            rwuserdata.h
        DESTINATION "${LIBRW_INSTALL_INCLUDEDIR}/src"
    )
    install(
        FILES
            d3d/rwxbox.h
            d3d/rwd3d.h
            d3d/rwd3d8.h
            d3d/rwd3d9.h
        DESTINATION "${LIBRW_INSTALL_INCLUDEDIR}/src/d3d"
    )
    install(
        FILES
            ps2/rwps2.h
            ps2/rwps2plg.h
        DESTINATION "${LIBRW_INSTALL_INCLUDEDIR}/src/ps2"
    )
    install(
        FILES
            gl/rwwdgl.h
            gl/rwgl3.h
            gl/rwgl3plg.h
            gl/rwgl3shader.h
        DESTINATION "${LIBRW_INSTALL_INCLUDEDIR}/src/gl"
    )

    install(
        FILES
            gl/glad/glad.h
            gl/glad/khrplatform.h
        DESTINATION "${LIBRW_INSTALL_INCLUDEDIR}/src/gl/glad"
    )

    install(
        TARGETS librw
        EXPORT librw-targets
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
endif()
