find_package(Doxygen)

if (DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/docs)
    configure_file(
            Doxyfile.in
            ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            @ONLY
    )

    add_custom_target(doxygen
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen..."
            VERBATIM
    )
else ()
    message(WARNING "Doxygen not found. The 'doxygen' target will not be created.")
endif ()