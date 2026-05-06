add_rules("mode.debug", "mode.release")

-- Option to enable/disable CUDA
option("cuda")
    set_default(false)
    set_showmenu(true)
    set_description("Enable CUDA support")
    on_check(function (option)
        if os.execv("nvidia-smi") == 0 then
            option:set_value(true)
        end
    end)
option_end()

-- Dependencies from xrepo
add_requires("sqlite3", "toml11", "openmp")
add_requires("imgui-sfml v3.0", {configs = {imgui = "v1.91.5"}})
add_requires("sfml", {configs = {audio = false, network = false}})

target("TicTacToe")
    set_kind("binary")
    
    if is_plat("linux") then
        add_syslinks("X11", "Xrandr", "Xcursor", "Xinerama", "Xi", "udev", "GL", "pthread", "dl")
    end
    
    on_load(function (target)
        -- Check if libtorch is present in the project directory
        local libtorch_dir = path.join(os.projectdir(), "libtorch")
        if not os.isdir(libtorch_dir) then
            print("LibTorch not found. Downloading...")
            local url
            if is_plat("linux") then
                if get_config("cuda") then
                    url = "https://download.pytorch.org/libtorch/cu124/libtorch-shared-with-deps-2.5.1%2Bcu124.zip"
                else
                    url = "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.5.1%2Bcpu.zip"
                end
            elseif is_plat("macosx") then
                url = "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.5.1.zip"
            end
            
            if url then
                os.runv("curl", {"-L", url, "-o", "libtorch.zip"})
                os.runv("unzip", {"-q", "libtorch.zip"})
                os.rm("libtorch.zip")
            else
                raise("Platform not supported for auto-download")
            end
        end
        
        target:add("includedirs", "libtorch/include")
        target:add("includedirs", "libtorch/include/torch/csrc/api/include")
        target:add("linkdirs", "libtorch/lib")
        target:add("links", "torch", "torch_cpu", "c10")
        
        if get_config("cuda") then
            target:add("links", "torch_cuda")
        end
        
        target:add("runenvs", "LD_LIBRARY_PATH", path.join(os.projectdir(), "libtorch/lib"))
        target:add("runenvs", "DYLD_LIBRARY_PATH", path.join(os.projectdir(), "libtorch/lib"))
    end)

    add_files("src/*.cpp")
    add_includedirs("include")
    add_packages("sqlite3", "toml11", "openmp", "imgui-sfml", "sfml")
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
