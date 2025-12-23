from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain, cmake_layout
from conan.errors import ConanInvalidConfiguration


class AlpmapConan(ConanFile):
    name = "alpmap"
    version = "0.1.0"
    settings = "os", "compiler", "build_type", "arch"

    def validate(self):
        """Ensure C++23 is used"""
        if self.settings.compiler.cppstd:
            cppstd = str(self.settings.compiler.cppstd)
            # Accept gnu23, 23, or higher
            if cppstd not in ["23", "gnu23"] and not cppstd.startswith("2"):
                raise ConanInvalidConfiguration("C++23 or higher required for modules")

    def requirements(self):
        self.requires("asio/1.30.2")
        self.requires("argparse/2.1")
        self.requires("gtest/1.14.0")
        self.requires("spdlog/1.15.3")
        self.requires("tomlplusplus/3.4.0")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        # Ensure C++23 in toolchain
        tc.variables["CMAKE_CXX_STANDARD"] = "23"
        tc.generate()
