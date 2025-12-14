from typing import TypeAlias

from generated.elantlr.EasyLanguageParser import EasyLanguageParser


ELDeclarationItemContent: TypeAlias = (
    EasyLanguageParser.Input_itemContext | EasyLanguageParser.Var_itemContext | EasyLanguageParser.Const_itemContext
)


def is_buy_order(ctx: EasyLanguageParser.Order_sideContext | None) -> bool:
    if ctx is None:
        return False

    return bool(ctx.BUY()) and not bool(ctx.BUYTOCOVER()) and not bool(ctx.COVER())


def is_sell_short_order(ctx: EasyLanguageParser.Order_sideContext | None) -> bool:
    if ctx is None:
        return False

    return bool(ctx.SELLSHORT()) or (bool(ctx.SELL()) and bool(ctx.SHORT()))


def is_sell_order(ctx: EasyLanguageParser.Order_sideContext | None) -> bool:
    if ctx is None:
        return False

    return bool(ctx.SELL()) and not bool(ctx.SELLSHORT()) and not bool(ctx.SHORT())


def is_buy_to_cover_order(ctx: EasyLanguageParser.Order_sideContext | None) -> bool:
    if ctx is None:
        return False

    return bool(ctx.BUYTOCOVER()) or (bool(ctx.BUY()) and bool(ctx.COVER()))
