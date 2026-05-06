add_rules("mode.debug", "mode.release")

-- Option to enable/disable CUDA
option("cuda")
    set_default(false)
    set_showmenu(true)
    set_description("Enable CUDA support")
    on_check(function (option)
        import("core.base.os")
        if os.execv("nvidia-smi") == 0 then
            option:set_value(true)
        end
    end)
option_end()

add_requires("sqlite3", "toml11", "openmp")

target("TicTacToe")
    set_kind("binary")
    add_files("src/*.cpp")
    add_includedirs("include")
    
    -- Use the libtorch directory (manually managed or symlinked)
    if os.isdir("libtorch") then
        add_includedirs("libtorch/include")
        add_includedirs("libtorch/include/torch/csrc/api/include")
        add_linkdirs("libtorch/lib")
        add_links("torch", "torch_cpu", "c10")
        if get_config("cuda") then
            add_links("torch_cuda")
        end
        
        add_runenvs("LD_LIBRARY_PATH", os.projectdir() .. "/libtorch/lib")
        add_runenvs("DYLD_LIBRARY_PATH", os.projectdir() .. "/libtorch/lib")
    end

    add_packages("sqlite3", "toml11", "openmp")
    set_languages("c++17")
    
    if is_plat("linux", "macosx") then
        add_syslinks("pthread", "dl")
    end

    if is_plat("linux") then
        add_defines("_GLIBCXX_USE_CXX11_ABI=0")
    end

    after_build(function (target)
        if os.isfile("config.toml") then
            os.cp("config.toml", target:targetdir())
        end
    end)
