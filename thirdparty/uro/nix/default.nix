{beamPackages, pkgs, ...}: with pkgs; beamPackages.mixRelease rec {
  pname = "uro";
  version = "0.0.1";

  inherit elixir;
  buildInputs = [ erlang ];

  src = ./..;

  mixFodDeps = beamPackages.fetchMixDeps {
    pname = "deps-${pname}";
    inherit src version;
    hash = "sha256-dAGug+GkNldlJHB/1Ir/6W6sVPZ0/ZuyR6JJsEsu6ac=";
  };

  postInstall = ''
    echo "DONE!"
  '';
}
