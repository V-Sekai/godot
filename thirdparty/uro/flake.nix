{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=24.11";
  };

  outputs = { self, nixpkgs }:
  let
    supportedSystems = [ "x86_64-linux" "aarch64-linux" ];
    forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    pkgs = forAllSystems (system: nixpkgs.legacyPackages.${system});

    packageFor = pkgs: rec {
      uro = pkgs.callPackage ./nix {};
      default = uro;
    };
  in 
    {
      devShells.x86_64-linux.default = pkgs.x86_64-linux.mkShell {
        buildInputs = with pkgs.x86_64-linux; [erlang_26 elixir_1_16 elixir-ls tailwindcss-language-server inotify-tools];
        shellHook = ''
          export HEX_OFFLINE=0
        '';
      };
      packages = forAllSystems (system:
        let packages = packageFor pkgs.${system}; in
        packages // { default = packages.default; });

      overlay = final: prev: packageFor final;
    };
}
