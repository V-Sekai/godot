﻿//using System.CommandLine;
//using System.CommandLine.Parsing;
//using Xunit;

//namespace Squirrel.CommandLine.Tests.Commands
//{
//    public class PackCommandTests : BaseCommandTests<PackOsxCommand>
//    {
//        [Fact]
//        public void Command_WithValidRequiredArguments_Parses()
//        {
//            DirectoryInfo packDir = CreateTempDirectory();
//            CreateTempFile(packDir);
//            var command = new PackCommand();

//            ParseResult parseResult = command.Parse($"--packId Clowd.Squirrel -v 1.2.3 -p \"{packDir.FullName}\"");

//            Assert.Empty(parseResult.Errors);
//            Assert.Equal("Clowd.Squirrel", command.PackId);
//            Assert.Equal("1.2.3", command.PackVersion);
//            Assert.Equal(packDir.FullName, command.PackDirectory?.FullName);
//        }

//        [Fact]
//        public void PackId_WithInvalidNuGetId_ShowsError()
//        {
//            DirectoryInfo packDir = CreateTempDirectory();
//            CreateTempFile(packDir);
//            var command = new PackCommand();

//            ParseResult parseResult = command.Parse($"--packId $42@ -v 1.0.0 -p \"{packDir.FullName}\"");

//            Assert.Equal(1, parseResult.Errors.Count);
//            Assert.StartsWith("--packId is an invalid NuGet package id.", parseResult.Errors[0].Message);
//            Assert.Contains("$42@", parseResult.Errors[0].Message);
//        }

//        [Fact]
//        public void PackVersion_WithInvalidVersion_ShowsError()
//        {
//            DirectoryInfo packDir = CreateTempDirectory();
//            CreateTempFile(packDir);
//            var command = new PackCommand();

//            ParseResult parseResult = command.Parse($"-u Clowd.Squirrel --packVersion 1.a.c -p \"{packDir.FullName}\"");

//            Assert.Equal(1, parseResult.Errors.Count);
//            Assert.StartsWith("--packVersion contains an invalid package version", parseResult.Errors[0].Message);
//            Assert.Contains("1.a.c", parseResult.Errors[0].Message);
//        }

//        [Fact]
//        public void PackDirectory_WithEmptyFolder_ShowsError()
//        {
//            DirectoryInfo packDir = CreateTempDirectory();
//            var command = new PackCommand();

//            ParseResult parseResult = command.Parse($"-u Clowd.Squirrel -v 1.0.0 --packDir \"{packDir.FullName}\"");

//            Assert.Equal(1, parseResult.Errors.Count);
//            Assert.StartsWith("--packDir must a non-empty directory", parseResult.Errors[0].Message);
//            Assert.Contains(packDir.FullName, parseResult.Errors[0].Message);
//        }

//        [Fact]
//        public void PackAuthors_WithMultipleAuthors_ParsesValue()
//        {
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + "--packAuthors Me,mysel,I";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.Equal("Me,mysel,I", command.PackAuthors);
//        }

//        [Fact]
//        public void PackTitle_WithTitle_ParsesValue()
//        {
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + "--packTitle \"My Awesome Title\"";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.Equal("My Awesome Title", command.PackTitle);
//        }

//        [Fact]
//        public void IncludePdb_BareOption_SetsFlag()
//        {
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + "--includePdb";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.True(command.IncludePdb);
//        }

//        [Fact]
//        public void ReleaseNotes_WithExistingFile_ParsesValue()
//        {
//            FileInfo releaseNotes = CreateTempFile();
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + $"--releaseNotes \"{releaseNotes.FullName}\"";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.Equal(releaseNotes.FullName, command.ReleaseNotes?.FullName);
//        }

//        [Fact]
//        public void ReleaseNotes_WithoutFile_ShowsError()
//        {
//            string releaseNotes = Path.GetFullPath(Path.GetRandomFileName());
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + $"--releaseNotes \"{releaseNotes}\"";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.Equal(1, parseResult.Errors.Count);
//            Assert.Equal(command.ReleaseNotes, parseResult.Errors[0].SymbolResult?.Symbol.Parents.Single());
//            Assert.Contains(releaseNotes, parseResult.Errors[0].Message);
//        }

//        [Fact]
//        public void SquirrelAwareExecutable_WithFileName_ParsesValue()
//        {
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + $"--mainExe \"MyApp.exe\"";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.Equal("MyApp.exe", command.SquirrelAwareExecutable);
//        }

//        [Fact]
//        public void Icon_WithValidFile_ParsesValue()
//        {
//            FileInfo fileInfo = CreateTempFile(name: Path.ChangeExtension(Path.GetRandomFileName(), ".ico"));
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + $"--icon \"{fileInfo.FullName}\"";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.Equal(fileInfo.FullName, command.Icon?.FullName);
//        }

//        [Fact]
//        public void Icon_WithBadFileExtension_ShowsError()
//        {
//            FileInfo fileInfo = CreateTempFile(name: Path.ChangeExtension(Path.GetRandomFileName(), ".wrong"));
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + $"--icon \"{fileInfo.FullName}\"";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.Equal(1, parseResult.Errors.Count);
//            Assert.Equal($"--icon does not have an .ico extension", parseResult.Errors[0].Message);
//        }

//        [Fact]
//        public void Icon_WithoutFile_ShowsError()
//        {
//            string file = Path.GetFullPath(Path.ChangeExtension(Path.GetRandomFileName(), ".ico"));
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + $"--icon \"{file}\"";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.Equal(1, parseResult.Errors.Count);
//            Assert.Equal(command.Icon, parseResult.Errors[0].SymbolResult?.Symbol.Parents.Single());
//            Assert.Contains(file, parseResult.Errors[0].Message);
//        }

//        [Fact]
//        public void BundleId_WithValue_ParsesValue()
//        {
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + $"--bundleId \"some id\"";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.Equal("some id", command.BundleId);
//        }

//        [Fact]
//        public void NoDelta_BareOption_SetsFlag()
//        {
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + "--noDelta";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.True(command.NoDelta);
//        }

//        [Fact]
//        public void NoPackage_BareOption_SetsFlag()
//        {
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + "--noPkg";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.True(command.NoPackage);
//        }

//        [Fact]
//        public void PackageContent_CanSpecifyMultipleValues()
//        {
//            DirectoryInfo packDir = CreateTempDirectory();
//            FileInfo testFile1 = CreateTempFile(packDir);
//            FileInfo testFile2 = CreateTempFile(packDir);
//            PackCommand command = new PackCommand();
//            string cli = $"-u clowd.squirrel -v 1.0.0 -p \"{packDir.FullName}\"";
//            cli += $" --pkgContent welcome={testFile1.FullName}";
//            cli += $" --pkgContent license={testFile2.FullName}";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.Empty(parseResult.Errors);
//            var packageContent = command.PackageContent;
//            Assert.Equal(2, packageContent?.Length);

//            Assert.Equal("welcome", packageContent![0].Key);
//            Assert.Equal(testFile1.FullName, packageContent![0].Value.FullName);

//            Assert.Equal("license", packageContent![1].Key);
//            Assert.Equal(testFile2.FullName, packageContent![1].Value.FullName);
//        }

//        [Fact]
//        public void PackageContent_WihtInvalidKey_DisplaysError()
//        {
//            DirectoryInfo packDir = CreateTempDirectory();
//            FileInfo testFile1 = CreateTempFile(packDir);
//            PackCommand command = new PackCommand();
//            string cli = $"-u clowd.squirrel -v 1.0.0 -p \"{packDir.FullName}\"";
//            cli += $" --pkgContent unknown={testFile1.FullName}";
//            ParseResult parseResult = command.Parse(cli);

//            ParseError error = parseResult.Errors.Single();
//            Assert.Equal("Invalid pkgContent key: unknown. Must be one of: welcome, readme, license, conclusion", error.Message);
//        }

//        [Fact]
//        public void SigningAppIdentity_WithSubject_ParsesValue()
//        {
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + $"--signAppIdentity \"Mac Developer\"";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.Equal("Mac Developer", command.SigningAppIdentity);
//        }

//        [Fact]
//        public void SigningInstallIdentity_WithSubject_ParsesValue()
//        {
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + $"--signInstallIdentity \"Mac Developer\"";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.Equal("Mac Developer", command.SigningInstallIdentity);
//        }

//        [Fact]
//        public void SigningEntitlements_WithValidFile_ParsesValue()
//        {
//            FileInfo fileInfo = CreateTempFile(name: Path.ChangeExtension(Path.GetRandomFileName(), ".entitlements"));
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + $"--signEntitlements \"{fileInfo.FullName}\"";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.Equal(fileInfo.FullName, command.SigningEntitlements?.FullName);
//        }

//        [Fact]
//        public void SigningEntitlements_WithBadFileExtension_ShowsError()
//        {
//            FileInfo fileInfo = CreateTempFile(name: Path.ChangeExtension(Path.GetRandomFileName(), ".wrong"));
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + $"--signEntitlements \"{fileInfo.FullName}\"";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.Equal(1, parseResult.Errors.Count);
//            Assert.Equal($"--signEntitlements does not have an .entitlements extension", parseResult.Errors[0].Message);
//        }

//        [Fact]
//        public void SigningEntitlements_WithoutFile_ShowsError()
//        {
//            string file = Path.GetFullPath(Path.ChangeExtension(Path.GetRandomFileName(), ".entitlements"));
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + $"--signEntitlements \"{file}\"";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.Equal(1, parseResult.Errors.Count);
//            Assert.Equal(command.SigningEntitlements, parseResult.Errors[0].SymbolResult?.Symbol.Parents.Single());
//            Assert.Contains(file, parseResult.Errors[0].Message);
//        }

//        [Fact]
//        public void NotaryProfile_WithName_ParsesValue()
//        {
//            var command = new PackCommand();

//            string cli = GetRequiredDefaultOptions() + $"--notaryProfile \"profile name\"";
//            ParseResult parseResult = command.Parse(cli);

//            Assert.Equal("profile name", command.NotaryProfile);
//        }

//        protected override string GetRequiredDefaultOptions()
//        {
//            DirectoryInfo packDir = CreateTempDirectory();
//            CreateTempFile(packDir);

//            return $"-u Clowd.Squirrel -v 1.0.0 -p \"{packDir.FullName}\" ";
//        }
//    }
//}
