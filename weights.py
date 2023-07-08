import numpy as np

weights = \
[
    # Conv 1
    [
        # Filter 1
        [
            [
                [
                    [
                        -0.15766790509223938,
                        -1.4090425968170166
                    ],
                    [
                        0.5764434933662415,
                        -0.18251268565654755
                    ]
                ],
                [
                    [
                        -0.4359842538833618,
                        1.2024283409118652
                    ],
                    [
                        -0.7871080636978149,
                        -0.20256073772907257
                    ]
                ]
            ],
            [
                [
                    [
                        1.0746972560882568,
                        0.7813117504119873
                    ],
                    [
                        0.8431252241134644,
                        0.48101720213890076
                    ]
                ],
                [
                    [
                        1.950129508972168,
                        1.4773359298706055
                    ],
                    [
                        1.391786813735962,
                        1.9511833190917969
                    ]
                ]
            ],
            [
                [
                    [
                        -0.631779670715332,
                        -0.728387713432312
                    ],
                    [
                        -0.19134916365146637,
                        -0.2982683777809143
                    ]
                ],
                [
                    [
                        -1.1117124557495117,
                        -0.9558441638946533
                    ],
                    [
                        -0.801497220993042,
                        -1.4129266738891602
                    ]
                ]
            ]
        ],
        # Filter 2
        [
            [
                [
                    [
                        -0.2359021008014679,
                        -0.5881462693214417
                    ],
                    [
                        -0.8138604164123535,
                        -0.17128989100456238
                    ]
                ],
                [
                    [
                        -0.8961703777313232,
                        1.2153191566467285
                    ],
                    [
                        -0.36075088381767273,
                        -1.475214958190918
                    ]
                ]
            ],
            [
                [
                    [
                        -1.00680410861969,
                        1.9569377899169922
                    ],
                    [
                        -0.8115332722663879,
                        -0.5544629693031311
                    ]
                ],
                [
                    [
                        -0.9461975693702698,
                        0.007296960800886154
                    ],
                    [
                        -0.8724292516708374,
                        -0.528706431388855
                    ]
                ]
            ],
            [
                [
                    [
                        0.8534883260726929,
                        -0.1022961288690567
                    ],
                    [
                        0.4329586923122406,
                        0.06024252250790596
                    ]
                ],
                [
                    [
                        -0.13578027486801147,
                        0.18875186145305634
                    ],
                    [
                        -1.7422243356704712,
                        0.7448899745941162
                    ]
                ]
            ]
        ],
        # Filter 3
        [
            [
                [
                    [
                        -1.0640618801116943,
                        -1.9751765727996826
                    ],
                    [
                        -0.7774884700775146,
                        0.09738417714834213
                    ]
                ],
                [
                    [
                        -0.6315925717353821,
                        -1.284070611000061
                    ],
                    [
                        -0.7543332576751709,
                        -0.19574621319770813
                    ]
                ]
            ],
            [
                [
                    [
                        1.7254054546356201,
                        1.7996954917907715
                    ],
                    [
                        0.25167620182037354,
                        -0.1816527098417282
                    ]
                ],
                [
                    [
                        0.9779609441757202,
                        -0.1319543719291687
                    ],
                    [
                        -0.11906915158033371,
                        0.6220499277114868
                    ]
                ]
            ],
            [
                [
                    [
                        -0.42461156845092773,
                        0.5572296380996704
                    ],
                    [
                        0.1722114235162735,
                        0.09834732860326767
                    ]
                ],
                [
                    [
                        -0.2371840626001358,
                        0.5913163423538208
                    ],
                    [
                        -0.010857663117349148,
                        -0.24695375561714172
                    ]
                ]
            ]
        ],
        # Filter 4
        [
            [
                [
                    [
                        0.2836855351924896,
                        0.4059520661830902
                    ],
                    [
                        0.5667827725410461,
                        0.31956300139427185
                    ]
                ],
                [
                    [
                        1.095702052116394,
                        1.6294035911560059
                    ],
                    [
                        0.5918745398521423,
                        1.9082716703414917
                    ]
                ]
            ],
            [
                [
                    [
                        -0.6465344429016113,
                        -1.2388174533843994
                    ],
                    [
                        -1.113878607749939,
                        -0.3532300591468811
                    ]
                ],
                [
                    [
                        -1.1922683715820312,
                        -2.170825958251953
                    ],
                    [
                        -0.6198663711547852,
                        -2.213268995285034
                    ]
                ]
            ],
            [
                [
                    [
                        0.3846951723098755,
                        0.7087291479110718
                    ],
                    [
                        0.7157859802246094,
                        0.06643503904342651
                    ]
                ],
                [
                    [
                        0.09043636918067932,
                        0.4955514371395111
                    ],
                    [
                        0.058077819645404816,
                        0.3869550824165344
                    ]
                ]
            ]
        ]
    ],
    [
        # Basis 1
        [
            [
                [
                    -0.39456242322921753
                ]
            ]
        ],
        # Basis 2
        [
            [
                [
                    -0.10785569995641708
                ]
            ]
        ],
        # Basis 3
        [
            [
                [
                    0.4411979615688324
                ]
            ]
        ],
        # Basis 4
        [
            [
                [
                    -0.07568704336881638
                ]
            ]
        ]
    ],
    # Conv 2
    [
        # Filter 1
        [
            [
                [
                    [
                        0.23204021155834198,
                        0.22960682213306427
                    ],
                    [
                        -0.05011353641748428,
                        0.607266366481781
                    ]
                ],
                [
                    [
                        0.9462418556213379,
                        1.6649869680404663
                    ],
                    [
                        0.37263134121894836,
                        0.6967059969902039
                    ]
                ]
            ],
            [
                [
                    [
                        0.05838308483362198,
                        -0.016460111364722252
                    ],
                    [
                        0.03562457114458084,
                        -0.08338908851146698
                    ]
                ],
                [
                    [
                        -0.5733880400657654,
                        0.34575942158699036
                    ],
                    [
                        -0.052727412432432175,
                        -0.7971541881561279
                    ]
                ]
            ],
            [
                [
                    [
                        0.4916303753852844,
                        -0.4432876408100128
                    ],
                    [
                        -0.5309923887252808,
                        0.903561532497406
                    ]
                ],
                [
                    [
                        0.0004726084880530834,
                        0.30845028162002563
                    ],
                    [
                        -1.4041754007339478,
                        -0.3247189223766327
                    ]
                ]
            ],
            [
                [
                    [
                        -1.8506306409835815,
                        -0.6528539061546326
                    ],
                    [
                        -0.8959575891494751,
                        0.2561679780483246
                    ]
                ],
                [
                    [
                        -1.6919327974319458,
                        -1.515560507774353
                    ],
                    [
                        -1.2092101573944092,
                        -1.216013789176941
                    ]
                ]
            ]
        ],
        # Filter 2
        [
            [
                [
                    [
                        1.0100398063659668,
                        0.8632034063339233
                    ],
                    [
                        0.3841843903064728,
                        -0.16572631895542145
                    ]
                ],
                [
                    [
                        1.0875825881958008,
                        -0.04117417335510254
                    ],
                    [
                        0.6507325768470764,
                        0.3937075436115265
                    ]
                ]
            ],
            [
                [
                    [
                        0.3459572196006775,
                        -0.8950223326683044
                    ],
                    [
                        0.007779011502861977,
                        -0.42971572279930115
                    ]
                ],
                [
                    [
                        0.5860533118247986,
                        -1.3510349988937378
                    ],
                    [
                        0.269628643989563,
                        -0.30277019739151
                    ]
                ]
            ],
            [
                [
                    [
                        -0.6275251507759094,
                        1.4693946838378906
                    ],
                    [
                        1.1012355089187622,
                        -0.4624863862991333
                    ]
                ],
                [
                    [
                        0.7786211967468262,
                        0.9392200112342834
                    ],
                    [
                        1.092366099357605,
                        -0.13469316065311432
                    ]
                ]
            ],
            [
                [
                    [
                        -0.7893413305282593,
                        -1.3000404834747314
                    ],
                    [
                        -0.4628649055957794,
                        -1.5523102283477783
                    ]
                ],
                [
                    [
                        -1.3639854192733765,
                        -1.4011399745941162
                    ],
                    [
                        -1.1600297689437866,
                        -0.6572083830833435
                    ]
                ]
            ]
        ]
    ],
    [
        # Basis 1
        [
            [
                [
                    0.47343772649765015
                ]
            ]
        ],
        # Basis 2
        [
            [
                [
                    0.7876431345939636
                ]
            ]
        ]
    ],
    # Conv 3
    [
        # Filter 1
        [
            [
                [
                    [
                        -0.41118964552879333
                    ]
                ]
            ],
            [
                [
                    [
                        0.2781473696231842
                    ]
                ]
            ]
        ]
    ],
    [
        # Basis 1
        [
            [
                [
                    -0.09273391962051392
                ]
            ]
        ]
    ]
]