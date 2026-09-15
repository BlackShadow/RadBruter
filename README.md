![RadBruter](Banner.png)

# RadBruter

RadBruter allows you to recover lightmap values from compiled **GoldSrc** maps.
RadBruter automatically detects maps original RAD compiler and bruteforces based on that. 
Additionally, detects for CUDA compatiable card. If CUDA is not an option for you, it can use CPU.
CUDA acceleration currently available for QRAD; other compilers use the CPU.

## Download

Get the **Windows x64 ZIP** from [Releases](https://github.com/BlackShadow/RadBruter/releases/latest).
Extract the whole archive and keep the executable, DLLs and `compilers` folder together.

## Usage

```bat
radbruter.exe map.bsp
```

On Linux, use `./radbruter`. The `.bsp` extension is optional.

After loading a map, choose **Fast**, **Balanced** (default), or **Full** recovery.
Live progress shows the map, compiler, completed work and an estimated remaining time.
The result is saved to `output/<map>.rad` with bruted values.

| Option | Purpose |
| --- | --- |
| `-o FILE` | Choose the output RAD file. |
| `--waddir DIR` | Supply external WAD textures. |
| `-qrad`, `-zhlt`, `-vhlt`, `-sdhlt` | Force a compiler instead of automatic selection. |
| `--quality fast\|balanced\|full` | Choose quality without the menu. |
| `--threads N` | Set the thread limit. |
| `--backend auto\|cpu\|cuda` | Choose the processing backend. |
| `-v`, `--verbose` | Show detailed diagnostics. |
| `-h`, `--help` | Show help. |

## Build

The first build needs internet access to restore dependencies automatically.

### Windows

Install **Visual Studio 2022 or newer**, the **Desktop development with C++** workload and a Windows SDK.
Open `RadBruter.sln`, select **Release | x64**, and build. CMake and the CUDA Toolkit are not required.

Or use a Visual Studio Developer terminal:

```bat
msbuild RadBruter.sln /m /p:Configuration=Release /p:Platform=x64
```

### Linux

Install GCC, **CMake 3.24+**, Ninja and OpenSSL development headers, then run:

```sh
cmake --preset release
cmake --build --preset release
```

Output: `build/Release/radbruter.exe` on Windows or `build/Release/radbruter` on Linux.
Use Release for normal recovery; Debug is much slower.

Compiler selection and recovered values are estimates, not proof of the original compiler or RAD file.
See [compiler details](src/COMPILERS.md).
