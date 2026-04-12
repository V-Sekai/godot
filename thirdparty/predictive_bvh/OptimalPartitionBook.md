## Generate Optimal Partition

The spatial oracle decides *who receives data* in a multiplayer game server. The core algorithm is an overlap-adaptive Hilbert broadphase with predictive expansion.

## Summary

Multiplayer game servers hit hard player-count ceilings because the spatial query that decides who receives data scales as O(N^2). The broadphase pairwise overlap check dominates every simulation step.

A game server can replace the O(N^2) broadphase with a proved-optimal O(N+k) Hilbert broadphase and predictive ghost expansion. The core algorithms are formally verified in Lean 4 and code-generated into the `predictive_bvh.h` header (R128 64.64 fixed-point, micrometres).

## Hilbert Broadphase: O(N+k)

1. **Radix sort** entities by 30-bit Hilbert code (Skilling 2004): O(N)
2. **Form groups** by Hilbert prefix: O(N)
3. **Prune** non-overlapping groups — `hilbert_prune_sound` (proved): if group AABBs don't overlap, no entity pairs between them can overlap
4. **Scan** overlapping groups: O(N+k), proportional to output

`broadphase_lower_bound` (proved in `Core/LowerBound.lean`) establishes Ω(N+k) — this is asymptotically optimal.

## Predictive Expansion

Each entity's bounding box is expanded by its velocity and acceleration to cover a rebuild window Δt of physics-bounded motion, using the Surface Area Heuristic [@macdonald1990bsp]:

$$e_\alpha = |V_\alpha| \cdot \Delta t + \tfrac{1}{2} A_\alpha \cdot \Delta t^2 \quad \text{(metres)}$$

`expansion_covers_k_ticks` (proved in `Core/Formula.lean`) guarantees no false negatives: any entity obeying Newtonian bounds stays inside its expanded box across the rebuild window. Per-entity Δt lets slow entities rebuild less often than fast ones:

| Entity type | Velocity | Rebuild window Δt | Ghost radius |
|---|---|---|---|
| Stationary spectator | 0 m/s | 6 s | 0 m |
| Concert dancer | 2 m/s | 0.65 s | ~1.6 m |
| Sprint infantry | 7 m/s | 0.2 s | ~2.1 m |
| Helicopter | 80 m/s | 0.05 s | ~4 m |

All algorithm logic is in Lean 4 [@demoura2021lean4]. Integer arithmetic throughout — `omega` closes every goal.

## Why This Matters

Multiplayer game servers hit hard player-count ceilings because the spatial query that decides who receives data scales poorly. The bottleneck is the broadphase — the pairwise overlap check that runs every tick.

This work replaces that bottleneck with a proved-optimal O(N+k) broadphase and a predictive expansion that lets the server skip rebuilds for slow-moving entities entirely across multiple simulation steps, with formal guarantees that no entity is ever missed.

## References

```bibtex
@misc{lambdaclass2024amo,
  author       = {{LambdaClass}},
  title        = {{AMO-Lean}: Towards Formally Verified Optimization via Equality Saturation in {Lean} 4},
  year         = {2024},
  url          = {https://blog.lambdaclass.com/amo-lean-towards-formally-verified-optimization-via-equality-saturation-in-lean-4/},
  urldate      = {2026-03-29}
}

@misc{truthresearch2024,
  author       = {{LambdaClass}},
  title        = {truth\_research\_zk: Efficient Equality Saturation with Union-Find in {Lean} 4},
  year         = {2024},
  url          = {https://github.com/lambdaclass/truth_research_zk}
}

@article{willsey2021egg,
  author  = {Willsey, Max and Nandi, Chandrakana and Wang, Yisu Remy and Flatt, Oliver and Tatlock, Zachary and Panchekha, Pavel},
  title   = {egg: Fast and Extensible Equality Saturation},
  journal = {Proceedings of the ACM on Programming Languages},
  volume  = {5},
  number  = {POPL},
  pages   = {1--29},
  year    = {2021},
  doi     = {10.1145/3434304}
}

@inproceedings{demoura2021lean4,
  author    = {de Moura, Leonardo and Ullrich, Sebastian},
  title     = {The {Lean} 4 Theorem Prover and Programming Language},
  booktitle = {Automated Deduction -- CADE 28},
  year      = {2021},
  doi       = {10.1007/978-3-030-79876-5_37}
}

@inproceedings{mathlib2020,
  author    = {{The mathlib Community}},
  title     = {The {Lean} Mathematical Library},
  booktitle = {Proceedings of the 9th ACM SIGPLAN International Conference on Certified Programs and Proofs},
  year      = {2020},
  doi       = {10.1145/3372885.3373824}
}

@article{macdonald1990bsp,
  author  = {MacDonald, J. David and Booth, Kellogg S.},
  title   = {Heuristics for Ray Tracing Using Space Subdivision},
  journal = {The Visual Computer},
  volume  = {6},
  number  = {3},
  pages   = {153--166},
  year    = {1990},
  doi     = {10.1007/BF01911006}
}

@inproceedings{wald2007fast,
  author    = {Wald, Ingo},
  title     = {On Fast Construction of {SAH}-Based Bounding Volume Hierarchies},
  booktitle = {2007 IEEE Symposium on Interactive Ray Tracing},
  pages     = {33--40},
  year      = {2007},
  doi       = {10.1109/RT.2007.4342588}
}

@inproceedings{karras2012maximizing,
  author    = {Karras, Tero},
  title     = {Maximizing Parallelism in the Construction of {BVH}s, Octrees, and k-d Trees},
  booktitle = {Proceedings of the Fourth ACM SIGGRAPH / Eurographics Conference on High-Performance Graphics},
  year      = {2012},
  doi       = {10.2312/EGGH/HPG12/033-037}
}

@inproceedings{chen2018av1,
  author    = {Chen, Yunqing and Murherjee, Debargha and Han, Jingning and Grange, Adrian and Xu, Yaowu and Liu, Zoe and Parker, Sarah and Chen, Cheng and Su, Hui and Joshi, Urvang and others},
  title     = {An Overview of Core Coding Tools in the {AV1} Video Codec},
  booktitle = {2018 Picture Coding Symposium (PCS)},
  year      = {2018},
  doi       = {10.1109/PCS.2018.8456249}
}

@misc{epicgames2022worldpartition,
  author       = {{Epic Games}},
  title        = {World Partition in {Unreal Engine} 5},
  year         = {2022},
  howpublished = {\url{https://docs.unrealengine.com/5.0/en-US/world-partition-in-unreal-engine/}}
}

@misc{unity2022scenestreaming,
  author       = {{Unity Technologies}},
  title        = {Scene Streaming and Addressable Assets},
  year         = {2022},
  howpublished = {\url{https://docs.unity3d.com/Packages/com.unity.addressables@1.21/manual/index.html}}
}

@misc{godot2024bvhstructs,
  author = {{Godot Engine contributors}},
  title  = {{Godot 4.x} Dynamic {BVH} Broadphase: node expansion constants},
  year   = {2024},
  url    = {https://github.com/godotengine/godot/blob/master/servers/physics_3d/godot_broad_phase_3d_bvh.h}
}

@book{Nystrom2014gpp,
  author    = {Nystrom, Robert},
  title     = {Game Programming Patterns},
  publisher = {Genever Benning},
  year      = {2014},
  isbn      = {978-0990582908}
}

@book{fabian2018dod,
  author    = {Fabian, Richard},
  title     = {Data-Oriented Design: Software Engineering for Limited Resources and Short Schedules},
  publisher = {Leanpub},
  year      = {2018},
  url       = {https://www.dataorienteddesign.com/dodbook/}
}

@inproceedings{lopez2018microscopic,
  author    = {Lopez, Pablo Alvarez and Behrisch, Michael and Bieker-Walz, Laura and Erdmann, Jakob and Fl{\"o}tter{\"o}d, Yun-Pang and Hilbrich, Robert and L{\"u}cken, Leonhard and Rummel, Johannes and Wagner, Peter and Wie{\ss}ner, Evamarie},
  title     = {Microscopic Traffic Simulation Using {SUMO}},
  booktitle = {The 21st IEEE International Conference on Intelligent Transportation Systems},
  year      = {2018},
  doi       = {10.1109/ITSC.2018.8569938}
}

@misc{sumo2024eclipse,
  author = {{Eclipse Foundation} and {DLR Institute of Transportation Systems}},
  title  = {{SUMO} -- Simulation of Urban {MObility}},
  year   = {2024},
  url    = {https://eclipse.dev/sumo/}
}

@inproceedings{lauterbach2009fast,
  author    = {Lauterbach, Christian and Garland, Michael and Sengupta, Shubhabrata and Luebke, David and Manocha, Dinesh},
  title     = {Fast {BVH} Construction on {GPU}s},
  booktitle = {Computer Graphics Forum},
  volume    = {28},
  number    = {2},
  pages     = {375--384},
  year      = {2009},
  doi       = {10.1111/j.1467-8659.2009.01377.x}
}

@misc{ps2guinness2022,
  author = {{Guinness World Records} and {Daybreak Game Company}},
  title  = {Most players simultaneously in a first-person shooter battle: {Planetside 2} world record},
  year   = {2022},
  url    = {https://www.guinnessworldrecords.com/world-records/most-players-simultaneously-in-a-fps-battle}
}

@misc{vrchat2026kaguya,
  author  = {{Road to VR}},
  title   = {{VRChat} Breaks All-Time Concurrent User Record with {Japanese Anime Concert}},
  year    = {2026},
  url     = {https://www.roadtovr.com/vrchat-all-time-high-japanese-concert/},
  urldate = {2026-03-30}
}

@misc{vrcschool2024networksync,
  author = {{VRChat School}},
  title  = {{VRChat} Network Sync and {IK} Update Rate},
  year   = {2024},
  url    = {https://vrchatschool.com/network-sync/}
}

@misc{photon2024regions,
  author = {{Exit Games}},
  title  = {{Photon PUN2}: Regions and Best Region},
  year   = {2024},
  url    = {https://doc.photonengine.com/pun/current/connection-and-authentication/regions}
}

@misc{resonitewiki2024arch,
  author = {{Resonite Wiki contributors}},
  title  = {Resonite Architecture and Server Infrastructure},
  year   = {2024},
  url    = {https://wiki.resonite.com/Server_Infrastructure}
}

@techreport{itu1998bt1359,
  author      = {{ITU-R}},
  title       = {Relative Timing of Sound and Vision for Broadcasting},
  institution = {International Telecommunication Union},
  type        = {Recommendation},
  number      = {BT.1359-1},
  year        = {1998}
}

@article{marsden1978servo,
  author  = {Marsden, C. D. and Merton, P. A. and Morton, H. B.},
  title   = {Servo action in the human thumb},
  journal = {The Journal of Physiology},
  volume  = {257},
  number  = {1},
  pages   = {1--44},
  year    = {1978},
  doi     = {10.1113/jphysiol.1978.sp011354}
}

@article{botvinick1998rubber,
  author  = {Botvinick, Matthew and Cohen, Jonathan},
  title   = {Rubber hands `feel' touch that eyes see},
  journal = {Nature},
  volume  = {391},
  pages   = {756},
  year    = {1998},
  doi     = {10.1038/35784}
}

@techreport{mills2010rfc5905,
  author      = {Mills, David and Martin, Juergen and Burbank, Jack and Kasch, William},
  title       = {Network Time Protocol Version 4: Protocol and Algorithms Specification},
  institution = {IETF},
  type        = {RFC},
  number      = {5905},
  year        = {2010},
  doi         = {10.17487/RFC5905}
}

@inproceedings{cronin2003cheatproof,
  author    = {Cronin, Eric and Filstrup, Brian and Jamin, Sugih},
  title     = {Cheat-Proofing Dead Reckoned Multiplayer Games},
  booktitle = {Workshop on Anti-Cheating in Online Games (ADCOG), co-located with ACM SIGCOMM},
  year      = {2003}
}

@article{liu2008hackproof,
  author    = {Liu, C. S. and Lui, John C. S.},
  title     = {Hack-Proof Synchronization Protocol for Multi-Player Online Games},
  journal   = {Multimedia Tools and Applications},
  year      = {2008},
  publisher = {Springer},
  doi       = {10.1007/s11042-008-0230-3}
}

@inproceedings{popov2009objectpartitioning,
  author    = {Popov, Stefan and Georgiev, Iliyan and Dimov, Rossen and Slusallek, Philipp},
  title     = {Object Partitioning Considered Harmful: Space Subdivision for {BVH}s},
  booktitle = {Proceedings of the 1st ACM Conference on High Performance Graphics ({HPG})},
  year      = {2009},
  publisher = {ACM}
}

@article{baughman2007cheatproof,
  author  = {Baughman, Nathaniel E. and Liberatore, Marc and Levine, Brian Neil},
  title   = {Cheat-Proof Playout for Centralized and Peer-to-Peer Gaming},
  journal = {{IEEE/ACM} Transactions on Networking},
  volume  = {15},
  number  = {1},
  year    = {2007},
  doi     = {10.1109/TNET.2006.886289}
}

@inproceedings{bharambe2004spectators,
  author    = {Bharambe, Ashwin R. and Padmanabhan, Venkata N. and Seshan, Srinivasan},
  title     = {Supporting Spectators in Online Multiplayer Games},
  booktitle = {Proceedings of the 3rd Workshop on Hot Topics in Networks ({HotNets}-{III})},
  year      = {2004},
  url       = {https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/hotnets2004.pdf}
}

@inproceedings{lee2018lagcompensation,
  author    = {Lee, Steven W. K. and Chang, Rocky K. C.},
  title     = {Enhancing the Experience of Multiplayer Shooter Games via Advanced Lag Compensation},
  booktitle = {Proceedings of the 9th {ACM} Multimedia Systems Conference},
  series    = {MMSys '18},
  pages     = {284--293},
  year      = {2018},
  doi       = {10.1145/3204949.3204971}
}

@article{liu2022survey,
  author    = {Liu, Shengmei and Xu, Xiaokun and Claypool, Mark},
  title     = {A Survey and Taxonomy of Latency Compensation Techniques for Network Computer Games},
  journal   = {ACM Computing Surveys},
  volume    = {54},
  number    = {11s},
  articleno = {243},
  year      = {2022},
  doi       = {10.1145/3519023}
}

@inproceedings{lazaridis2021hitboxes,
  author    = {Lazaridis, Lazaros and Papatsimouli, Maria and Kollias, Konstantinos F. and Sarigiannidis, Panagiotis and Fragulis, George F.},
  title     = {Hitboxes: A Survey About Collision Detection in Video Games},
  booktitle = {{HCI} in Games: Experience Design and Game Mechanics},
  series    = {Lecture Notes in Computer Science},
  volume    = {12789},
  pages     = {307--320},
  publisher = {Springer, Cham},
  year      = {2021},
  doi       = {10.1007/978-3-030-77277-2_24}
}

@article{muller2007pbd,
  author  = {M{\"u}ller, Matthias and Heidelberger, Bruno and Hennix, Marcus and Ratcliff, John},
  title   = {Position Based Dynamics},
  journal = {Journal of Visual Communication and Image Representation},
  volume  = {18},
  number  = {2},
  pages   = {109--118},
  year    = {2007},
  doi     = {10.1016/j.jvcir.2007.01.005}
}

@inproceedings{ahmed2008dynAoI,
  author    = {Ahmed, Dewan T. and Shirmohammadi, Shervin},
  title     = {A Dynamic Area of Interest Management and Collaboration Model for {P2P} {MMOGs}},
  booktitle = {Proceedings of the 12th {IEEE/ACM} International Symposium on Distributed Simulation and Real-Time Applications},
  series    = {DS-RT '08},
  pages     = {27--34},
  year      = {2008},
  doi       = {10.1109/DS-RT.2008.26}
}

@article{liu2014interestmgmt,
  author    = {Liu, Elvis S. and Theodoropoulos, Georgios K.},
  title     = {Interest Management for Distributed Virtual Environments: A Survey},
  journal   = {ACM Computing Surveys},
  volume    = {46},
  number    = {4},
  articleno = {51},
  pages     = {51:1--51:42},
  year      = {2014},
  doi       = {10.1145/2535417}
}

@inproceedings{boulanger2006comparing,
  author    = {Boulanger, Jean-S{\'e}bastien and Kienzle, J{\"o}rg and Verbrugge, Clark},
  title     = {Comparing Interest Management Algorithms for Massively Multiplayer Games},
  booktitle = {Proceedings of the 5th {ACM} {SIGCOMM} Workshop on Network and System Support for Games},
  series    = {NetGames '06},
  articleno = {6},
  year      = {2006},
  doi       = {10.1145/1230040.1230046}
}

@inproceedings{morgan2004expandingspheres,
  author    = {Morgan, Graham and Storey, Kier and Lu, Fengyun},
  title     = {Expanding Spheres: A Collision Detection Algorithm for Interest Management in Networked Games},
  booktitle = {Entertainment Computing -- {ICEC} 2004},
  series    = {Lecture Notes in Computer Science},
  volume    = {3166},
  pages     = {507--516},
  publisher = {Springer, Berlin, Heidelberg},
  year      = {2004},
  doi       = {10.1007/978-3-540-28643-1_56}
}

@article{benor1983algebraic,
  author  = {Ben-Or, Michael},
  title   = {Lower Bounds for Algebraic Computation Trees},
  journal = {Proceedings of the Fifteenth Annual ACM Symposium on Theory of Computing},
  pages   = {80--86},
  year    = {1983},
  doi     = {10.1145/800061.808735}
}

@inproceedings{basch1999kinetic,
  author    = {Basch, Julien and Guibas, Leonidas J. and Hershberger, John},
  title     = {Data Structures for Mobile Data},
  booktitle = {Journal of Algorithms},
  volume    = {31},
  number    = {1},
  pages     = {1--28},
  year      = {1999},
  doi       = {10.1006/jagm.1998.0988}
}

@inproceedings{fdb2021paper,
  author    = {Zhou, Jingyu and Xu, Meng and Shraer, Alexander and Namasivayam, Bala and Miller, Alex and Tschannen, Evan and Atherton, Steve and Beez, Andrew J. and Hammer, Pat and Arefin, Soumya and Banning, Jared and Laber, Markus and Stumm, Michael and Butterstein, David and Fischel, Brian and Gazit, Benny and Terber, Avi and Collins, Rusty},
  title     = {{FoundationDB}: A Distributed Unbundled Transactional Key Value Store},
  booktitle = {Proceedings of the 2021 International Conference on Management of Data (SIGMOD)},
  year      = {2021},
  doi       = {10.1145/3448016.3457559}
}

@misc{cockroachdb2020parallelcommits,
  author = {{Cockroach Labs}},
  title  = {Parallel Commits {TLA+} Specification},
  year   = {2020},
  url    = {https://github.com/v-sekai/cockroach/blob/release-22.1-oxide/docs/tla-plus/ParallelCommits/ParallelCommits.tla}
}

@inproceedings{kulkarni2014hlc,
  author    = {Kulkarni, Sandeep and Demirbas, Murat and Madappa, Deepak and Avva, Bharadwaj and Leone, Marcelo},
  title     = {Logical Physical Clocks and Consistent Snapshots in Globally Distributed Databases},
  booktitle = {Stabilization, Safety, and Security of Distributed Systems (SSS 2014)},
  year      = {2014},
  doi       = {10.1007/978-3-319-11764-5_14}
}

@misc{enet2024,
  author = {Lee Salzman},
  title  = {{ENet}: Reliable {UDP} Networking Library},
  year   = {2024},
  url    = {http://enet.bespin.org/}
}

@article{Liu2014InterestManagement,
  author    = {Elvis S. Liu and Georgios K. Theodoropoulos},
  title     = {Interest Management for Distributed Virtual Environments: A Survey},
  journal   = {ACM Computing Surveys},
  volume    = {46},
  number    = {4},
  pages     = {51:1--51:42},
  year      = {2014},
  doi       = {10.1145/2535417},
}

@incollection{Ahmed2009ZoningAOI,
  author    = {D. T. Ahmed and S. Shirmohammadi},
  title     = {Zoning Issues and Area of Interest Management in Massively Multiplayer Online Games},
  booktitle = {Handbook of Multimedia for Digital Entertainment and Arts},
  publisher = {Springer},
  address   = {Boston, MA},
  year      = {2009},
  doi       = {10.1007/978-0-387-89024-1_8},
}

@inproceedings{Boulanger2006ComparingIM,
  author    = {Jean-S{\'e}bastien Boulanger and J{\"o}rg Kienzle and Clark Verbrugge},
  title     = {Comparing Interest Management Algorithms for Massively Multiplayer Games},
  booktitle = {Proceedings of the 5th ACM SIGCOMM Workshop on Network and System Support for Games (NetGames '06)},
  year      = {2006},
  doi       = {10.1145/1230040.1230069},
}

@article{Beskow2009PartialMigration,
  author    = {Paul B. Beskow and Knut-Helge Vik and P{\aa}l Halvorsen and Carsten Griwodz},
  title     = {The Partial Migration of Game State and Dynamic Server Selection to Reduce Latency},
  journal   = {Multimedia Tools and Applications},
  volume    = {45},
  number    = {1--3},
  pages     = {83--107},
  year      = {2009},
  doi       = {10.1007/s11042-009-0287-7},
}

@inproceedings{Jaya2016IMDeadReckoning,
  author    = {Iryanto Jaya and others},
  title     = {Combining Interest Management and Dead Reckoning: A Hybrid Approach for Efficient Data Distribution in Multiplayer Online Games},
  booktitle = {IEEE Conference on Games and Entertainment},
  year      = {2016},
  doi       = {10.1109/7789876},
}

@book{bader2013sfc,
  author    = {Bader, Michael},
  title     = {Space-Filling Curves: An Introduction with Applications in Scientific Computing},
  publisher = {Springer},
  series    = {Texts in Computational Science and Engineering},
  volume    = {9},
  year      = {2013},
  doi       = {10.1007/978-3-642-31046-1}
}

@inproceedings{skilling2004hilbert,
  author    = {Skilling, John},
  title     = {Programming the {Hilbert} Curve},
  booktitle = {AIP Conference Proceedings},
  volume    = {707},
  pages     = {381--387},
  year      = {2004},
  doi       = {10.1063/1.1751381}
}

@techreport{hamilton2006compact,
  author      = {Hamilton, Chris H. and Rau-Chaplin, Andrew},
  title       = {Compact {Hilbert} Indices},
  institution = {Dalhousie University, Faculty of Computer Science},
  type        = {Technical Report},
  number      = {CS-2006-07},
  year        = {2006}
}

@inproceedings{jacobson1988congestion,
  author    = {Jacobson, Van and Karels, Michael J.},
  title     = {Congestion Avoidance and Control},
  booktitle = {Proceedings of {SIGCOMM} '88},
  pages     = {314--329},
  year      = {1988},
  doi       = {10.1145/52324.52356}
}

@inproceedings{karn1987rtt,
  author    = {Karn, Phil and Partridge, Craig},
  title     = {Improving Round-Trip Time Estimates in Reliable Transport Protocols},
  booktitle = {Proceedings of {ACM SIGCOMM} '87},
  pages     = {2--7},
  year      = {1987},
  doi       = {10.1145/55483.55484}
}

@inproceedings{braud2021talaria,
  author    = {Braud, Tristan and Alhilal, Ahmad and Hui, Pan},
  title     = {Talaria: In-Engine Synchronisation for Seamless Migration of Mobile Edge Gaming Instances},
  booktitle = {Proceedings of the 17th International Conference on Emerging Networking EXperiments and Technologies ({CoNEXT} '21)},
  pages     = {375--381},
  year      = {2021},
  doi       = {10.1145/3485983.3494848}
}
```
