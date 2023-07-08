import numpy as np

weights = \
[
    #Conv 1
    [
        # Filter 1
        [
            [
                [
                    [
                        0.0529952272772789,
                        0.07766587287187576
                    ],
                    [
                        0.06479708850383759,
                        0.08849441260099411
                    ]
                ],
                [
                    [
                        0.15089112520217896,
                        0.11827833950519562
                    ],
                    [
                        0.15612711012363434,
                        0.12259481102228165
                    ]
                ]
            ],
            [
                [
                    [
                        -0.10967392474412918,
                        -0.14659425616264343
                    ],
                    [
                        -0.07855850458145142,
                        -0.11950015276670456
                    ]
                ],
                [
                    [
                        -0.24305754899978638,
                        -0.23457486927509308
                    ],
                    [
                        -0.2848691940307617,
                        -0.2653895616531372
                    ]
                ]
            ],
            [
                [
                    [
                        0.09328152239322662,
                        0.10509538650512695
                    ],
                    [
                        0.07647393643856049,
                        0.07021910697221756
                    ]
                ],
                [
                    [
                        0.1288672238588333,
                        0.12797896564006805
                    ],
                    [
                        0.14394953846931458,
                        0.16809774935245514
                    ]
                ]
            ]
        ],
        # Filter 2
        [
            [
                [
                    [
                        0.034149955958127975,
                        -0.10252805054187775
                    ],
                    [
                        -0.029102589935064316,
                        -0.12048713117837906
                    ]
                ],
                [
                    [
                        -0.0008648376679047942,
                        -0.0750395655632019
                    ],
                    [
                        -0.06963378190994263,
                        -0.1672588437795639
                    ]
                ]
            ],
            [
                [
                    [
                        0.017312098294496536,
                        -0.27797263860702515
                    ],
                    [
                        0.06143740192055702,
                        -0.24076882004737854
                    ]
                ],
                [
                    [
                        0.055566031485795975,
                        -0.2678932845592499
                    ],
                    [
                        0.05243650823831558,
                        -0.28997907042503357
                    ]
                ]
            ],
            [
                [
                    [
                        -0.058738503605127335,
                        -0.08435320854187012
                    ],
                    [
                        -0.09139306098222733,
                        -0.07726815342903137
                    ]
                ],
                [
                    [
                        -0.07652799785137177,
                        -0.06692185252904892
                    ],
                    [
                        -0.09269090741872787,
                        -0.1380058079957962
                    ]
                ]
            ]
        ],
        # Filter 3
        [
            [
                [
                    [
                        0.08093049377202988,
                        0.04442744329571724
                    ],
                    [
                        -0.06869158148765564,
                        0.03749972954392433
                    ]
                ],
                [
                    [
                        -0.0015058942371979356,
                        -0.04351396486163139
                    ],
                    [
                        -0.21830271184444427,
                        -0.04494459927082062
                    ]
                ]
            ],
            [
                [
                    [
                        -0.2570515275001526,
                        -0.17522583901882172
                    ],
                    [
                        -0.6939678192138672,
                        -0.2315361499786377
                    ]
                ],
                [
                    [
                        0.15744762122631073,
                        0.21107693016529083
                    ],
                    [
                        -0.37355169653892517,
                        0.1593305617570877
                    ]
                ]
            ],
            [
                [
                    [
                        0.08611151576042175,
                        0.06797682493925095
                    ],
                    [
                        0.030620058998465538,
                        0.13558684289455414
                    ]
                ],
                [
                    [
                        -0.12727363407611847,
                        -0.11777037382125854
                    ],
                    [
                        -0.18263405561447144,
                        -0.05371926724910736
                    ]
                ]
            ]
        ],
        # Filter 4
        [
            [
                [
                    [
                        -0.18268238008022308,
                        0.004942948464304209
                    ],
                    [
                        0.029095441102981567,
                        -0.0018372217891737819
                    ]
                ],
                [
                    [
                        -0.25910475850105286,
                        -0.0326070636510849
                    ],
                    [
                        -0.07986708730459213,
                        -0.020083189010620117
                    ]
                ]
            ],
            [
                [
                    [
                        -0.34410855174064636,
                        0.10297372192144394
                    ],
                    [
                        0.148189976811409,
                        0.13434018194675446
                    ]
                ],
                [
                    [
                        -0.46612533926963806,
                        -0.012896778993308544
                    ],
                    [
                        0.006893943063914776,
                        0.0773756206035614
                    ]
                ]
            ],
            [
                [
                    [
                        -0.09356722980737686,
                        -0.07457015663385391
                    ],
                    [
                        -0.05706440284848213,
                        -0.07126231491565704
                    ]
                ],
                [
                    [
                        -0.11058317869901657,
                        -0.03642761707305908
                    ],
                    [
                        -0.06675002723932266,
                        -0.08370325714349747
                    ]
                ]
            ]
        ]
    ],
    [
        # Bias 1
        [
            [
                [
                    -1.2860736846923828
                ]
            ]
        ],
        # Bias 2
        [
            [
                [
                    0.05457440763711929
                ]
            ]
        ],
        # Bias 3
        [
            [
                [
                    0.2608175277709961
                ]
            ]
        ],
        # Bias 4
        [
            [
                [
                    0.08029606938362122
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
                        -0.07999376952648163,
                        0.06024264916777611
                    ],
                    [
                        -0.028197776526212692,
                        0.06731494516134262
                    ]
                ],
                [
                    [
                        -0.20744241774082184,
                        -0.2320679873228073
                    ],
                    [
                        -0.11674708127975464,
                        -0.12800240516662598
                    ]
                ]
            ],
            [
                [
                    [
                        -0.1616075485944748,
                        -0.11818147450685501
                    ],
                    [
                        -0.08859618008136749,
                        -0.10779624432325363
                    ]
                ],
                [
                    [
                        -0.14233620464801788,
                        -0.17258816957473755
                    ],
                    [
                        -0.07504817843437195,
                        -0.07941270619630814
                    ]
                ]
            ],
            [
                [
                    [
                        -0.16732734441757202,
                        -0.17518864572048187
                    ],
                    [
                        -0.1105727106332779,
                        -0.12136920541524887
                    ]
                ],
                [
                    [
                        -0.07275930047035217,
                        -0.19998589158058167
                    ],
                    [
                        -0.07816190272569656,
                        -0.1151675432920456
                    ]
                ]
            ],
            [
                [
                    [
                        -0.10845547169446945,
                        -0.07114703208208084
                    ],
                    [
                        -0.07991690188646317,
                        -0.0845581591129303
                    ]
                ],
                [
                    [
                        -0.10557480156421661,
                        -0.1917378157377243
                    ],
                    [
                        -0.11026063561439514,
                        -0.13189753890037537
                    ]
                ]
            ]
        ],
        # Filter 2
        [
            [
                [
                    [
                        -0.36186763644218445,
                        -0.5053256750106812
                    ],
                    [
                        -0.21588240563869476,
                        -0.24873562157154083
                    ]
                ],
                [
                    [
                        -0.497949481010437,
                        -0.5219592452049255
                    ],
                    [
                        -0.3098538815975189,
                        -0.3074685335159302
                    ]
                ]
            ],
            [
                [
                    [
                        -0.08883466571569443,
                        -0.07451235502958298
                    ],
                    [
                        -0.029266124591231346,
                        -0.06171920895576477
                    ]
                ],
                [
                    [
                        -0.09632384032011032,
                        -0.1680031567811966
                    ],
                    [
                        -0.05504296347498894,
                        -0.08227800577878952
                    ]
                ]
            ],
            [
                [
                    [
                        -0.19166219234466553,
                        -0.1764293611049652
                    ],
                    [
                        -0.10313752293586731,
                        -0.10316595435142517
                    ]
                ],
                [
                    [
                        -0.005723975598812103,
                        -0.004154772497713566
                    ],
                    [
                        -0.008824986405670643,
                        0.026444219052791595
                    ]
                ]
            ],
            [
                [
                    [
                        -0.035497356206178665,
                        -0.024693559855222702
                    ],
                    [
                        -0.008623885922133923,
                        0.02128208987414837
                    ]
                ],
                [
                    [
                        -0.12449982762336731,
                        -0.11820479482412338
                    ],
                    [
                        -0.056399062275886536,
                        -0.10705026984214783
                    ]
                ]
            ]
        ]
    ],
    [
        # Bais 1
        [
            [
                [
                    0.048306968063116074
                ]
            ]
        ],
        # Bais 2
        [
            [
                [
                    -0.018294574692845345
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
                        -0.4018886983394623
                    ]
                ]
            ],
            [
                [
                    [
                        -0.553264319896698
                    ]
                ]
            ]
        ]
    ],
    [
        # Bias 1
        [
            [
                [
                    -0.08959515392780304
                ]
            ]
        ]
    ]
]