#include <stdio.h>
#include "../desync_c_interface.h"

int main() {
    int result = DesyncUntar("https://v-sekai.github.io/casync-v-sekai-game/store", 
                "https://github.com/V-Sekai/casync-v-sekai-game/raw/main/vsekai_game_windows_x86_64.caidx",
                "vsekai_game_windows_x86_64",
                "");
    if (result != 0) {
        printf("Error: storeUrl, indexUrl, and outputDir are required\n");
    }
    return 0;
}