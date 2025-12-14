from typing import Callable, Optional, TypeVar, cast

import antlr4
from antlr4 import ParserRuleContext
from antlr4.Parser import TerminalNode

from el_ast import CodeLocation
from generated.elantlr.EasyLanguageLexer import EasyLanguageLexer
from generated.elantlr.EasyLanguageParser import EasyLanguageParser
from generated.elantlr.EasyLanguageVisitor import EasyLanguageVisitor


def _token_index(tok: object | None, *, method_name: str, attr_name: str) -> Optional[int]:
    if tok is None:
        return None

    if hasattr(tok, method_name):
        try:
            return int(getattr(tok, method_name)())
        except Exception:
            return None

    if hasattr(tok, attr_name):
        try:
            return int(getattr(tok, attr_name))
        except Exception:
            return None

    return None


def token_start_index(tok: object | None) -> Optional[int]:
    return _token_index(tok, method_name="getStartIndex", attr_name="start")


def token_stop_index(tok: object | None) -> Optional[int]:
    return _token_index(tok, method_name="getStopIndex", attr_name="stop")


def ctx_start_token(ctx: ParserRuleContext | None) -> object | None:
    if ctx is None:
        return None

    if hasattr(ctx, "getStart"):
        try:
            return getattr(ctx, "getStart")()
        except Exception:
            pass

    return getattr(ctx, "start", None)


def ctx_stop_token(ctx: ParserRuleContext | None) -> object | None:
    if ctx is None:
        return None

    if hasattr(ctx, "getStop"):
        try:
            return getattr(ctx, "getStop")()
        except Exception:
            pass

    return getattr(ctx, "stop", None)


def _slice_source(code: str, start: Optional[int], end: Optional[int]) -> str:
    if start is None or end is None:
        return ""

    if start < 0 or end < 0 or start >= len(code) or end >= len(code) or start > end:
        return ""

    return code[start : end + 1]


class AntlrParserObjects:
    def __init__(self, input_stream: antlr4.InputStream):
        self.input = input_stream
        self.lexer = EasyLanguageLexer(self.input)
        self.tokens = antlr4.CommonTokenStream(self.lexer)
        self.parser = EasyLanguageParser(self.tokens)

    @staticmethod
    def from_string(code: str) -> "AntlrParserObjects":
        return AntlrParserObjects(antlr4.InputStream(code))

    def loc_from(self, obj: object) -> CodeLocation:
        if obj is None:
            return CodeLocation()

        if isinstance(obj, TerminalNode) or hasattr(obj, "getSymbol"):
            get_symbol = getattr(obj, "getSymbol", None)
            sym = get_symbol() if callable(get_symbol) else getattr(obj, "symbol", None)
            s = token_start_index(sym)
            e = token_stop_index(sym)
            if s is None or e is None:
                return CodeLocation()

            return CodeLocation(s, e)

        if not isinstance(obj, ParserRuleContext):
            return CodeLocation()

        start_tok = ctx_start_token(obj)
        stop_tok = ctx_stop_token(obj)
        s = token_start_index(start_tok)
        e = token_stop_index(stop_tok)
        if s is None or e is None:
            return CodeLocation()

        return CodeLocation(s, e)


CtxT = TypeVar("CtxT", bound=ParserRuleContext)
RuleHandler = Callable[[ParserRuleContext], None]
TerminalHandler = Callable[[TerminalNode], None]


class ELTreeWalker(EasyLanguageVisitor):
    def __init__(self):
        super().__init__()
        self._rule_handlers: dict[type[ParserRuleContext], list[RuleHandler]] = {}
        self._terminal_handlers: list[TerminalHandler] = []

    def register_handler(self, context_type: type[object], fn: Callable[[object], None]) -> None:
        if context_type is TerminalNode or issubclass(context_type, TerminalNode):
            self._terminal_handlers.append(cast(TerminalHandler, fn))
            return

        self._rule_handlers.setdefault(cast(type[ParserRuleContext], context_type), []).append(cast(RuleHandler, fn))

    def register_ctx_handler(self, context_type: type[CtxT], fn: Callable[[CtxT], None]) -> None:
        self.register_handler(context_type, cast(Callable[[object], None], fn))

    def register_terminal_handler(self, fn: TerminalHandler) -> None:
        self._terminal_handlers.append(fn)

    def walk(self, tree: ParserRuleContext) -> object:
        return self.visit(tree)

    def visitChildren(self, node: ParserRuleContext) -> object:
        try:
            handlers = self._rule_handlers.get(type(node))
            if handlers:
                for h in handlers:
                    h(node)
        except Exception:
            pass

        return super().visitChildren(node)

    def visitTerminal(self, node: TerminalNode) -> object:
        for h in self._terminal_handlers:
            h(node)

        return super().visitTerminal(node)


def get_source_text_ctx(ctx: ParserRuleContext | None, code: str) -> str:
    if ctx is None:
        return ""

    start_tok = ctx_start_token(ctx)
    stop_tok = ctx_stop_token(ctx)
    s = token_start_index(start_tok)
    e = token_stop_index(stop_tok)
    return _slice_source(code, s, e)


def get_source_text_terminal(node: TerminalNode | None, code: str) -> str:
    if node is None:
        return ""

    sym = getattr(node, "getSymbol")() if hasattr(node, "getSymbol") else getattr(node, "symbol", None)
    s = token_start_index(sym)
    e = token_stop_index(sym)
    return _slice_source(code, s, e)


def get_outermost_parent_scope(ctx: ParserRuleContext) -> ParserRuleContext:
    parent = ctx
    while parent is not None:
        p = getattr(parent, "parentCtx", None)
        if p is None:
            p = getattr(parent, "parent", None)

        if p is None:
            break

        if not isinstance(p, ParserRuleContext):
            break

        if isinstance(p, EasyLanguageParser.ProgramContext):
            break

        parent = p

    return parent
