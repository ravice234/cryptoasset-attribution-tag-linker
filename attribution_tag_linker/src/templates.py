# System
# System Extended
# Task
# Domain
# Question
# Output Format
# Output Format Reminder

SYSTEM_MESSAGE = "You are an AI assistant that follows instruction extremely well."
SYSTEM_MESSAGE_EXT = "You are an AI assistant that follows instruction extremely well. User will give you a question. Your task is to answer as faithfully as you can."
QUESTION = "Does the attribution tag refer to any of the listed entities and, if so, to which one?"

EXAMPLES = [

    {"input_string":
"""
Attribution Tag Label: WatchMyBit.com
[0] WatchMyBit.com
[1] MyBitcoin.com
[2] Buybitcoin.ph
[3] Hanbitco
[4] BitcoinWallet.com
[5] None of the entities above
""",
    "output": "0"},

    {"input_string":
"""
Attribution Tag Label: compound_Pair_ETH_USDC
[0] Compound
[1] Morpho Aave/Compound
[2] Tether (USDT)
[3] BitLaunder.com
[4] CryptoBounty.com
[5] None of the entities above
""",
    "output": "0"},

    {"input_string":
"""
Attribution Tag Label: Btc.top
[0] BTC.com
[1] BTCCPool
[2] 50BTC.com
[3] viabtc.com
[4] BTCGuild.com
[5] None of the entities above,
""",
      "output": "5"},

    {"input_string":
"""
Attribution Tag Label: curvefinance_UST-StableSwapUST
[0] TrustSwap
[1] Curve Finance
[2] PancakeSwap (AMM, v2, Stableswap) PancakeSwap StableSwap
[3] Zyberswap (AMM,Stableswap)
[4] StableKoi
[5] None of the entities above
""",
    "output": "1"},

    {"input_string":
"""
Attribution Tag Label: fei_Tribe
[0] weBribe DAO
[1] Atrix
[2] Fei Protocol
[3] Strike
[4] UniTrade
[5] None of the entities above
""",
    "output": "2"}
]

# All
T0 = [SYSTEM_MESSAGE_EXT,
"""
Determine if the attribution tag label refers to any of the listed entities and, if so, to which one.
The attribution tag describes an entity that is related to a blockchain address.
{input_string}
Does the attribution tag refer to any of the listed entities and, if so, to which one?
Choose your answer from: [0, 1, 2, 3, 4, 5]. Before answering, make sure your answer only contains a number."""]

# No Sys Extended
T1 = [SYSTEM_MESSAGE, 
"""Determine if the attribution tag label refers to any of the listed entities and, if so, to which one.
The attribution tag describes an entity that is related to a blockchain address.
{input_string}
Does the attribution tag refer to any of the listed entities and, if so, to which one?
Choose your answer from: [0, 1, 2, 3, 4, 5]. Before answering, make sure your answer only contains a number."""]

#  No output format reminder
T2 = [SYSTEM_MESSAGE_EXT,
"""Determine if the attribution tag label refers to any of the listed entities and, if so, to which one.
The attribution tag describes an entity that is related to a blockchain address.
{input_string}
Does the attribution tag refer to any of the listed entities and, if so, to which one?
Choose your answer from: [0, 1, 2, 3, 4, 5]."""]

# No Domain
T3 = [SYSTEM_MESSAGE_EXT,
"""Determine if the attribution tag label refers to any of the listed entities and, if so, to which one.
{input_string}
Does the attribution tag refer to any of the listed entities and, if so, to which one?
Choose your answer from: [0, 1, 2, 3, 4, 5]. Before answering, make sure your answer only contains a number."""]

# No output format nor reminder
T4 = [SYSTEM_MESSAGE_EXT,
"""Determine if the attribution tag label refers to any of the listed entities and, if so, to which one.
The attribution tag describes an entity that is related to a blockchain address.
{input_string}
Does the attribution tag refer to any of the listed entities and, if so, to which one?"""]

# No system 
T5 = ['',
"""Determine if the attribution tag label refers to any of the listed entities and, if so, to which one.
The attribution tag describes an entity that is related to a blockchain address.
{input_string}
Does the attribution tag refer to any of the listed entities and, if so, to which one?
Choose your answer from: [0, 1, 2, 3, 4, 5]. Before answering, make sure your answer only contains a number."""]

# No system extended, No domain, No outputformat reminder
T6 = ['',
"""You are an AI assistant that follows instruction extremely well. User will give you a question.
Determine if the attribution tag label refers to any of the listed entities and, if so, to which one.
{input_string}
Does the attribution tag refer to any of the listed entities and, if so, to which one?
Choose your answer from: [0, 1, 2, 3, 4, 5]."""]

# No system, no domain 
T7 = ['',
"""Determine if the attribution tag label refers to any of the listed entities and, if so, to which one.
{input_string}
Does the attribution tag refer to any of the listed entities and, if so, to which one?
Choose your answer from: [0, 1, 2, 3, 4, 5]. Before answering, make sure your answer only contains a number."""]

# No system extended, No domain, No outputformat nor reminder
T8 = ['',
"""You are an AI assistant that follows instruction extremely well. User will give you a question.
Determine if the attribution tag label refers to any of the listed entities and, if so, to which one.
{input_string}
Does the attribution tag refer to any of the listed entities and, if so, to which one?"""]

# No system, no domain, no output nor reminder
T9 = ['',
"""Determine if the attribution tag label refers to any of the listed entities and, if so, to which one.
{input_string}
Does the attribution tag refer to any of the listed entities and, if so, to which one?"""]

TEMPLATES = [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9]