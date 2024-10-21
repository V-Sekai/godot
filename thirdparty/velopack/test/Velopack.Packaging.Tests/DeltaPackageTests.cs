﻿//using System;
//using System.Collections.Generic;
//using System.IO;
//using System.Linq;
//using System.Threading.Tasks;
//using Squirrel.NuGet;
//using Squirrel;
//using Squirrel.SimpleSplat;
//using Squirrel.Tests.TestHelpers;
//using Xunit;
//using Xunit.Abstractions;
//using NuGet.Versioning;
//using Squirrel.CommandLine;

//namespace Squirrel.Tests
//{
//    public class ApplyDeltaPackageTests : TestLoggingBase
//    {
//        public ApplyDeltaPackageTests(ITestOutputHelper log) : base(log)
//        {
//        }

//        [Fact]
//        public void ApplyDeltaPackageSmokeTest()
//        {
//            var basePackage = IntegrationTestHelper.GetPath("fixtures", "Squirrel.Core.1.0.0.0-full.nupkg");
//            var deltaPackage = IntegrationTestHelper.GetPath("fixtures", "Squirrel.Core.1.1.0.0-delta.nupkg");
//            var expectedPackageFile = IntegrationTestHelper.GetPath("fixtures", "Squirrel.Core.1.1.0.0-full.nupkg");
//            var outFile = Path.GetTempFileName() + ".nupkg";

//            try {
//                var deltaBuilder = new DeltaPackage();
//                deltaBuilder.ApplyDeltaPackage(basePackage, deltaPackage, outFile);

//                var result = new ZipPackage(outFile);
//                var expected = new ZipPackage(expectedPackageFile);

//                result.Id.ShouldEqual(expected.Id);
//                result.Version.ShouldEqual(expected.Version);

//                this.Log().Info("Expected file list:");
//                var expectedList = expected.Files.Select(x => x.Path).OrderBy(x => x).ToList();
//                expectedList.ForEach(x => this.Log().Info(x));

//                this.Log().Info("Actual file list:");
//                var actualList = result.Files.Select(x => x.Path).OrderBy(x => x).ToList();
//                actualList.ForEach(x => this.Log().Info(x));

//                Enumerable.Zip(expectedList, actualList, (e, a) => e == a)
//                    .All(x => x != false)
//                    .ShouldBeTrue();
//            } finally {
//                if (File.Exists(outFile)) {
//                    File.Delete(outFile);
//                }
//            }
//        }

//        [Fact]
//        public void ApplyDeltaWithBothBsdiffAndNormalDiffDoesntFail()
//        {
//            var basePackage = IntegrationTestHelper.GetPath("fixtures", "slack-1.1.8-full.nupkg");
//            var deltaPackage = IntegrationTestHelper.GetPath("fixtures", "slack-1.2.0-delta.nupkg");
//            var outFile = Path.GetTempFileName() + ".nupkg";

//            try {
//                var deltaBuilder = new DeltaPackage();
//                deltaBuilder.ApplyDeltaPackage(basePackage, deltaPackage, outFile);

//                var result = new ZipPackage(outFile);

//                result.Id.ShouldEqual("slack");
//                result.Version.ShouldEqual(SemanticVersion.Parse("1.2.0"));
//            } finally {
//                if (File.Exists(outFile)) {
//                    File.Delete(outFile);
//                }
//            }
//        }
        
//        [Fact]
//        public void ApplyMultipleDeltasFast()
//        {
//            var basePackage = IntegrationTestHelper.GetPath("fixtures", "Clowd-3.4.287-full.nupkg");
//            var deltaPackage1 = IntegrationTestHelper.GetPath("fixtures", "Clowd-3.4.288-delta.nupkg");
//            var deltaPackage2 = IntegrationTestHelper.GetPath("fixtures", "Clowd-3.4.291-delta.nupkg");
//            var deltaPackage3 = IntegrationTestHelper.GetPath("fixtures", "Clowd-3.4.292-delta.nupkg");

//            using var t1 = Utility.GetTempDirectory(out var appDir);
//            using var t2 = Utility.GetTempDirectory(out var updateDir);

//            using var um = UpdateManagerTestImpl.FromLocalPackageTempDir(updateDir, "theApp", appDir);
//            var pkgDir = um.Config.PackagesDir;
            
//            File.Copy(basePackage, Path.Combine(pkgDir, Path.GetFileName(basePackage)));
//            File.Copy(deltaPackage1, Path.Combine(pkgDir, Path.GetFileName(deltaPackage1)));
//            File.Copy(deltaPackage2, Path.Combine(pkgDir, Path.GetFileName(deltaPackage2)));
//            File.Copy(deltaPackage3, Path.Combine(pkgDir, Path.GetFileName(deltaPackage3)));
            
//            var baseEntry = ReleaseEntry.GenerateFromFile(basePackage);

//            var toApply = new [] {
//                ReleaseEntry.GenerateFromFile(deltaPackage1),
//                ReleaseEntry.GenerateFromFile(deltaPackage2),
//                ReleaseEntry.GenerateFromFile(deltaPackage3),
//            };

//            List<int> progress = new List<int>();

//            var newEntry = um.createFullPackagesFromDeltas(toApply, baseEntry, progress.Add);

//            var outFile = Path.Combine(pkgDir, newEntry.Filename);
//            var result = new ZipPackage(outFile);
//            result.Id.ShouldEqual("Clowd");
//            result.Version.ShouldEqual(SemanticVersion.Parse("3.4.292"));
//        }

//        [Fact(Skip = "Rewrite this test, the original uses too many heavyweight fixtures")]
//        public void ApplyMultipleDeltaPackagesGeneratesCorrectHash()
//        {
//            Assert.Fail("Rewrite this test, the original uses too many heavyweight fixtures");
//        }
//    }

//    public class CreateDeltaPackageTests : IEnableLogger
//    {
//        [Fact]
//        public void CreateDeltaPackageIntegrationTest()
//        {
//            var basePackage = IntegrationTestHelper.GetPath("fixtures", "Squirrel.Tests.0.1.0-pre.nupkg");
//            var newPackage = IntegrationTestHelper.GetPath("fixtures", "Squirrel.Tests.0.2.0-pre.nupkg");

//            var baseFixture = new ReleasePackageBuilder(basePackage);
//            var fixture = new ReleasePackageBuilder(newPackage);

//            var tempFiles = Enumerable.Range(0, 3)
//                .Select(_ => Path.GetTempPath() + Guid.NewGuid().ToString() + ".nupkg")
//                .ToArray();

//            try {
//                baseFixture.CreateReleasePackage(tempFiles[0]);
//                fixture.CreateReleasePackage(tempFiles[1]);

//                (new FileInfo(baseFixture.ReleasePackageFile)).Exists.ShouldBeTrue();
//                (new FileInfo(fixture.ReleasePackageFile)).Exists.ShouldBeTrue();

//                var deltaBuilder = new DeltaPackageBuilder();
//                deltaBuilder.CreateDeltaPackage(baseFixture, fixture, tempFiles[2]);

//                var fullPkg = new ZipPackage(tempFiles[1]);
//                var deltaPkg = new ZipPackage(tempFiles[2]);

//                //
//                // Package Checks
//                //

//                fullPkg.Id.ShouldEqual(deltaPkg.Id);
//                fullPkg.Version.CompareTo(deltaPkg.Version).ShouldEqual(0);

//                // Delta packages should be smaller than the original!
//                var fileInfos = tempFiles.Select(x => new FileInfo(x)).ToArray();
//                this.Log().Info("Base Size: {0}, Current Size: {1}, Delta Size: {2}",
//                    fileInfos[0].Length, fileInfos[1].Length, fileInfos[2].Length);

//                (fileInfos[2].Length - fileInfos[1].Length).ShouldBeLessThan(0);

//                //
//                // File Checks
//                ///

//                var deltaPkgFiles = deltaPkg.Files.ToList();
//                deltaPkgFiles.Count.ShouldBeGreaterThan(0);

//                this.Log().Info("Files in delta package:");
//                deltaPkgFiles.ForEach(x => this.Log().Info(x.Path));

//                var newFilesAdded = new[] {
//                    "Newtonsoft.Json.dll",
//                    //"Refit.dll",
//                    //"Refit-Portable.dll",
//                    //"Castle.Core.dll",
//                }.Select(x => x.ToLowerInvariant());

//                // vNext adds a dependency on Refit
//                newFilesAdded
//                    .All(x => deltaPkgFiles.Any(y => y.Path.ToLowerInvariant().Contains(x)))
//                    .ShouldBeTrue();

//                // All the other files should be diffs and shasums
//                deltaPkgFiles
//                    .Where(x => !newFilesAdded.Any(y => x.Path.ToLowerInvariant().Contains(y)))
//                    .All(x => x.Path.ToLowerInvariant().EndsWith("bsdiff") || x.Path.ToLowerInvariant().EndsWith("shasum"))
//                    .ShouldBeTrue();

//                // Every .diff file should have a shasum file
//                deltaPkg.Files.Any(x => x.Path.ToLowerInvariant().EndsWith(".bsdiff")).ShouldBeTrue();
//                deltaPkg.Files
//                    .Where(x => x.Path.ToLowerInvariant().EndsWith(".bsdiff"))
//                    .ForEach(x => {
//                        var lookingFor = x.Path.Replace(".bsdiff", ".shasum");
//                        this.Log().Info("Looking for corresponding shasum file: {0}", lookingFor);
//                        deltaPkg.Files.Any(y => y.Path == lookingFor).ShouldBeTrue();
//                    });
//            } finally {
//                tempFiles.ForEach(File.Delete);
//            }
//        }

//        [Fact]
//        public void WhenBasePackageIsNewerThanNewPackageThrowException()
//        {
//            var basePackage = IntegrationTestHelper.GetPath("fixtures", "Squirrel.Tests.0.2.0-pre.nupkg");
//            var newPackage = IntegrationTestHelper.GetPath("fixtures", "Squirrel.Tests.0.1.0-pre.nupkg");

//            var baseFixture = new ReleasePackageBuilder(basePackage);
//            var fixture = new ReleasePackageBuilder(newPackage);

//            var tempFiles = Enumerable.Range(0, 3)
//                .Select(_ => Path.GetTempPath() + Guid.NewGuid().ToString() + ".nupkg")
//                .ToArray();

//            try {
//                baseFixture.CreateReleasePackage(tempFiles[0]);
//                fixture.CreateReleasePackage(tempFiles[1]);

//                (new FileInfo(baseFixture.ReleasePackageFile)).Exists.ShouldBeTrue();
//                (new FileInfo(fixture.ReleasePackageFile)).Exists.ShouldBeTrue();

//                Assert.Throws<InvalidOperationException>(() => {
//                    var deltaBuilder = new DeltaPackageBuilder();
//                    deltaBuilder.CreateDeltaPackage(baseFixture, fixture, tempFiles[2]);
//                });
//            } finally {
//                tempFiles.ForEach(File.Delete);
//            }
//        }

//        [Fact]
//        public void WhenBasePackageReleaseIsNullThrowsException()
//        {
//            var basePackage = IntegrationTestHelper.GetPath("fixtures", "Squirrel.Core.1.0.0.0.nupkg");
//            var newPackage = IntegrationTestHelper.GetPath("fixtures", "Squirrel.Core.1.1.0.0.nupkg");

//            var sourceDir = IntegrationTestHelper.GetPath("fixtures", "packages");
//            (new DirectoryInfo(sourceDir)).Exists.ShouldBeTrue();

//            var baseFixture = new ReleasePackageBuilder(basePackage);
//            var fixture = new ReleasePackageBuilder(newPackage);

//            var tempFile = Path.GetTempPath() + Guid.NewGuid() + ".nupkg";

//            try {
//                Assert.Throws<ArgumentException>(() => {
//                    var deltaBuilder = new DeltaPackageBuilder();
//                    deltaBuilder.CreateDeltaPackage(baseFixture, fixture, tempFile);
//                });
//            } finally {
//                File.Delete(tempFile);
//            }
//        }

//        [Fact]
//        public void WhenBasePackageDoesNotExistThrowException()
//        {
//            var basePackage = IntegrationTestHelper.GetPath("fixtures", "Squirrel.Tests.0.1.0-pre.nupkg");
//            var newPackage = IntegrationTestHelper.GetPath("fixtures", "Squirrel.Tests.0.2.0-pre.nupkg");

//            var baseFixture = new ReleasePackageBuilder(basePackage);
//            var fixture = new ReleasePackageBuilder(newPackage);

//            var tempFiles = Enumerable.Range(0, 3)
//                .Select(_ => Path.GetTempPath() + Guid.NewGuid().ToString() + ".nupkg")
//                .ToArray();

//            try {
//                baseFixture.CreateReleasePackage(tempFiles[0]);
//                fixture.CreateReleasePackage(tempFiles[1]);

//                (new FileInfo(baseFixture.ReleasePackageFile)).Exists.ShouldBeTrue();
//                (new FileInfo(fixture.ReleasePackageFile)).Exists.ShouldBeTrue();

//                // NOW WATCH AS THE FILE DISAPPEARS
//                File.Delete(baseFixture.ReleasePackageFile);

//                Assert.Throws<FileNotFoundException>(() => {
//                    var deltaBuilder = new DeltaPackageBuilder();
//                    deltaBuilder.CreateDeltaPackage(baseFixture, fixture, tempFiles[2]);
//                });
//            } finally {
//                tempFiles.ForEach(File.Delete);
//            }
//        }

//        [Fact]
//        public void WhenNewPackageDoesNotExistThrowException()
//        {
//            var basePackage = IntegrationTestHelper.GetPath("fixtures", "Squirrel.Tests.0.1.0-pre.nupkg");
//            var newPackage = IntegrationTestHelper.GetPath("fixtures", "Squirrel.Tests.0.2.0-pre.nupkg");

//            var baseFixture = new ReleasePackageBuilder(basePackage);
//            var fixture = new ReleasePackageBuilder(newPackage);

//            var tempFiles = Enumerable.Range(0, 3)
//                .Select(_ => Path.GetTempPath() + Guid.NewGuid().ToString() + ".nupkg")
//                .ToArray();

//            try {
//                baseFixture.CreateReleasePackage(tempFiles[0]);
//                fixture.CreateReleasePackage(tempFiles[1]);

//                (new FileInfo(baseFixture.ReleasePackageFile)).Exists.ShouldBeTrue();
//                (new FileInfo(fixture.ReleasePackageFile)).Exists.ShouldBeTrue();

//                // NOW WATCH AS THE FILE DISAPPEARS
//                File.Delete(fixture.ReleasePackageFile);

//                Assert.Throws<FileNotFoundException>(() => {
//                    var deltaBuilder = new DeltaPackageBuilder();
//                    deltaBuilder.CreateDeltaPackage(baseFixture, fixture, tempFiles[2]);
//                });
//            } finally {
//                tempFiles.ForEach(File.Delete);
//            }
//        }

//        [Fact]
//        public void HandleBsDiffWithoutExtraData()
//        {
//            var baseFileData = new byte[] { 1, 1, 1, 1 };
//            var newFileData = new byte[] { 2, 1, 1, 1 };

//            byte[] patchData;

//            using (var patchOut = new MemoryStream()) {
//                Bsdiff.BinaryPatchUtility.Create(baseFileData, newFileData, patchOut);
//                patchData = patchOut.ToArray();
//            }

//            using (var toPatch = new MemoryStream(baseFileData))
//            using (var patched = new MemoryStream()) {
//                Bsdiff.BinaryPatchUtility.Apply(toPatch, () => new MemoryStream(patchData), patched);

//                Assert.Equal(newFileData, patched.ToArray());
//            }
//        }
//    }
//}
