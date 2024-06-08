# godot-sentry-native

This is a custom C++ Godot Module that reports crashes (and other events) to [Sentry.io](https://sentry.io/).

This is super handy when trying to track down crashes that happen in release builds on your users devices.

So far this has only been tested on Windows 10 using Godot 4.0 rc1 64 bit and Sentry-native 0.6.0.

Sentry-native also supports macOS, Linux, and Android. So this module should be usable on those platforms with minor tweaking of the instructions below.

# Setup

This is a bit tricky to setup and get the folder structure right. After following the next couple of steps your folder structure should look like:

```
+-- godot
|   +-- bin
+-- modules/sentry
|   +-- example
|   +-- modules
|   +-- thirdparty
|       +-- sentry-native
```

## Clone and Compile Godot

First things first, you need to be able to build Godot from source
See [Compiling](https://docs.godotengine.org/en/stable/development/compiling/index.html) in Godot Docs

Continue only once you've got Godot compiling and running. For the rest of the instructions it's assumed that you're building Godot 3.4 32 bit on Windows.

## Clone this repo

1. git clone https://github.com/V-Sekai/godot-sentry-native
1. Or download and extract zip https://github.com/V-Sekai/godot-sentry-native/archive/refs/heads/master.zip

## Download Sentry Source Code

1. Download latest release (Currently 0.6.0) of sentry-native.zip from https://github.com/getsentry/sentry-native/releases
1. Extract to `thirdparty/sentry-native`

## Build Sentry dll with mingw-llvm

We need the sentry.dll and sentry.lib built before building godot-sentry-native

1. `cd thirdparty/sentry-native`
1. Run: `cmake -B build -DSENTRY_BACKEND=breakpad -DSENTRY_BUILD_SHARED_LIBS=NO -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSENTRY_BUILD_EXAMPLES=off -G"MinGW Makefiles"`
1. Run: `cd build && mingw32-make.exe -j8`

## Build Godot editor with godot-sentry-native with mingw-llvm

This is to build Godot editor for testing integration

1. cd to your godot folder
1. Run: `scons target=release_debug custom_modules=../modules use_llvm=yes use_mingw=yes LINKFLAGS='-Wl,-pdb=' CCFLAGS='-Wall -Wno-tautological-compare -g -gcodeview'`
1. If everything is successful, you should have: `godot/bin/godot.windows.opt.tools.exe`
1. `scoop install sentry-cli`
1. Upload debug symbols. 
1. `sentry-cli debug-files upload bin --org godot --project godot-project` Use `--wait` if you want to see errors.

## Create a Sentry.io account

Sentry.io is a hosted service that listens for crashes from your game, saves them, then presents them in a pretty dashboard. It's free for the first 5,000 crashes per month. Or you can run your own sentry server if you prefer.

1. Create an account at [Sentry.io](https://sentry.io/)
1. Walk through the onboarding process
   - Choose your project's platform
     1. Click on **Desktop**
     1. Click **Native**
     1. Click **Create Project**
   - Prepare the Native SDK
     - It will show you a chunk of C code, the only thing we need from it is the dsn
     - It looks like this: https://1234567890abcd@01234567.ingest.sentry.io/1234567
     - Leave your browser on this page, Note in the corner of the page it says **Waiting to receive first event to continue**

## Test it out with the example project

1. Run your custom build of Godot: `godot/bin/godot.windows.opt.tools.exe`
1. Import the example project: `godot-sentry-native/example/project.godot`
1. Click **Project** -> **Project Settings**
1. At the bottom of the list look for **Third Party** and click **Sentry**
1. Set Dsn to the url you copied from Sentry.io
1. Click Close
1. Click **Play(F5)** to run Example project
1. Click **Crash**
1. Note the Sentry.io page should now say **Event was received!** with a **Take me to my error** button. Click that.
1. You should now see a pretty report all about the crash

## Upload debug symbols

This step is optional, but highly recommended.
If you look at a crash's raw call stack you'll notice some lines that look like:
`godot.windows.opt.tools.32.exe      0x41ccccd           <unknown>`

That isn't very helpful, luckily we can get sentry.io to replace `<unknown>` with the name of the function by uploading godot's pdb symbol file to sentry.io

Here's how:

1. Download the latest release of Sentry Cli for Windows from: https://github.com/getsentry/sentry-cli/releases/
   - 32-bit: sentry-cli-Windows-i686.exe
   - 64-bit: sentry-cli-Windows-x86_64.exe
1. Copy and rename it to: `godot\bin\sentry-cli.exe`
1. cd to `godot\bin\`
1. Login to Sentry.io: `sentry-cli login`
   1. **Open browser now? [y/n]** y
   1. This will open a browser window, hopefully using the same browser you already logged into Sentry.io with.
   1. Click **Create New Token**
   1. The default Scopes should be fine. Click **Create Token**
   1. Copy the Auth Token and Paste into your cmd window prompt **Enter Your Token:**
1. Verify it's working: `sentry-cli info`
1. Get your Oranization Slug
   1. On Sentry.io Click on your name in the upper left Corner
   1. Click **Organization settings**
   1. Copy the value for **Organization Slug**
1. Get your Project Name
   1. Underneath **General Settings** click **Projects**
   1. Copy the project name, likely it is the same as your Organization Slug
1. Now upload symbol file: `sentry-cli upload-dif -o your-organization -p your-project --wait godot.windows.opt.tools.32.pdb `
   - Replace `your-organization` with your Organization Slug copied from website
   - Replace `your-project` with your Project Name copied from website
1. Then run the Example again and click Crash. Go look at the new crash on sentry.io and you should now see function names in the call stacks instead of `<unknown>`

## Building a new Godot Export Template

So far we've got the Godot editor to report crashes, but what we really want is for our packaged released games to report crashes happening to real users. To do that we need to build a new Godot Export Template.

1. See Godot Docs: [Creating Windows export templates](https://docs.godotengine.org/en/stable/development/compiling/compiling_for_windows.html#creating-windows-export-templates)
1. Just add `custom_modules=../godot-sentry-native/modules/sentry` to any of the scons commands
1. For example: `scons platform=windows tools=no target=release bits=32 custom_modules=../godot-sentry-native/modules/sentry`
1. Be sure to copy `godot/bin/sentry.dll` and `godot/bin/crashpad_handler.exe` to your game's export folder
1. Don't forget to upload symbol file any time you rebuild: `sentry-cli upload-dif -o your-organization -p your-project --wait godot.windows.opt.32.pdb`

## Setup Sentry.io in your own Godot Game Project's

To have your own games report crashes to Sentry.io:

1. Set Sentry Dsn in Project Settings
1. Export using the Custom Export Template
1. Be sure to copy `godot/bin/sentry.dll` and `godot/bin/crashpad_handler.exe` to your Export folder and ship with your game.

## Options in Project Settings

- String **thirdparty/sentry/dsn** \
   Dsn for your Sentry.io account
- String **thirdparty/sentry/app_version** \
   Version of your game, you should increment everytime you make a release. So that you can tell if a crash is coming from the current or old version.
- bool **thirdparty/sentry/send_ip_address** \
   Include ip address of user in crash report. Handy to tell if multiple crashes are coming from the same user or multiple users.

## Usage

Sentry is a singleton that will be available in all your gd script files

- You can send an event to Sentry.io at any time with: `Sentry.send_event("Test Event!")` \
   This can be handy for reporting errors other then crashes.
- Set username of user: `Sentry.set_username("Bob")`
