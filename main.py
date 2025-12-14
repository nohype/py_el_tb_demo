from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Generic, Iterable, Optional, TypeVar, Union, cast

import antlr4

from generated.elantlr.EasyLanguageLexer import EasyLanguageLexer
from generated.elantlr.EasyLanguageParser import EasyLanguageParser
from generated.elantlr.EasyLanguageVisitor import EasyLanguageVisitor

try:
    from antlr4 import ParserRuleContext
except Exception:

    class ParserRuleContext:  # type: ignore[no-redef]
        pass


try:
    from antlr4.tree.Tree import TerminalNode
except Exception:

    class TerminalNode:  # type: ignore[no-redef]
        pass


def _to_lower(s: str) -> str:
    return s.lower()


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


def _token_start_index(tok: object | None) -> Optional[int]:
    return _token_index(tok, method_name="getStartIndex", attr_name="start")


def _token_stop_index(tok: object | None) -> Optional[int]:
    return _token_index(tok, method_name="getStopIndex", attr_name="stop")


def _ctx_start_token(ctx: ParserRuleContext | None) -> object | None:
    if ctx is None:
        return None
    if hasattr(ctx, "getStart"):
        try:
            return ctx.getStart()
        except Exception:
            pass
    return getattr(ctx, "start", None)


def _ctx_stop_token(ctx: ParserRuleContext | None) -> object | None:
    if ctx is None:
        return None
    if hasattr(ctx, "getStop"):
        try:
            return ctx.getStop()
        except Exception:
            pass
    return getattr(ctx, "stop", None)


def _slice_source(code: str, s: Optional[int], e: Optional[int]) -> str:
    if s is None or e is None:
        return ""
    if s < 0 or e < 0 or s >= len(code) or e >= len(code) or s > e:
        return ""
    return code[s : e + 1]


@dataclass(order=True)
class CodeLocation:
    firstPos: int = -1
    lastPos: int = -1

    def is_valid(self) -> bool:
        return self.firstPos >= 0 and self.lastPos >= 0 and self.firstPos <= self.lastPos

    def length(self) -> int:
        return self.lastPos - self.firstPos + 1


@dataclass
class CodeBlock:
    loc: CodeLocation
    code: str


class OrderDirection(Enum):
    LongEntry = 1
    ShortEntry = 2
    LongExit = 3
    ShortExit = 4


class SlPtDirective(Enum):
    SetStopLoss = 1
    SetProfitTarget = 2
    SetContractOrPosition = 3


T = TypeVar("T")


class _CodeLocMapStruct(Generic[T]):
    def __init__(self, key_getter: Callable[[T], str]):
        self._key_getter = key_getter
        self._items: list[T] = []
        self._key_to_index: dict[str, int] = {}

    def add(self, item: T, replace_existing_key: bool = True) -> bool:
        key = _to_lower(self._key_getter(item))
        if replace_existing_key or key not in self._key_to_index:
            self._items.append(item)
            self._sort_and_index(erased=False)
            return True
        return False

    def add_many(self, items: Iterable[T], replace_existing_keys: bool = True) -> bool:
        ok = True
        for item in items:
            ok = self.add(item, replace_existing_key=replace_existing_keys) and ok
        return ok

    def clear(self) -> None:
        self._items.clear()
        self._key_to_index.clear()

    def empty(self) -> bool:
        return len(self._items) == 0

    def count(self) -> int:
        return len(self._items)

    def contains(self, key: str) -> bool:
        return _to_lower(key) in self._key_to_index

    def keys(self) -> list[str]:
        return list(self._key_to_index.keys())

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, key: Union[int, str]) -> T:
        if isinstance(key, int):
            return self._items[key]
        return self._items[self._key_to_index[_to_lower(key)]]

    def _sort_and_index(self, erased: bool) -> None:
        if erased:
            self._key_to_index.clear()
        self._items.sort(key=lambda x: getattr(x, "loc", CodeLocation()).firstPos)
        for i, item in enumerate(self._items):
            self._key_to_index[_to_lower(self._key_getter(item))] = i


@dataclass
class Variable:
    dataType: str = ""
    name: str = ""
    value: str = ""
    loc: CodeLocation = field(default_factory=CodeLocation)


class VariableDeclBlock(_CodeLocMapStruct[Variable]):
    def __init__(self):
        super().__init__(key_getter=lambda v: v.name)
        self.loc = CodeLocation()


@dataclass
class CaseBlock:
    loc: CodeLocation = field(default_factory=CodeLocation)
    caseValue: str = ""
    code: str = ""


class SwitchBlock(_CodeLocMapStruct[CaseBlock]):
    def __init__(self):
        super().__init__(key_getter=lambda c: c.caseValue)
        self.loc = CodeLocation()
        self.switchVar = ""


class SwitchBlockContainer(_CodeLocMapStruct[SwitchBlock]):
    def __init__(self):
        super().__init__(key_getter=lambda s: s.switchVar)


def join_code_blocks(blocks: list[CodeBlock], delim: str = "\n\n") -> str:
    return delim.join(b.code for b in blocks)


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
            s = _token_start_index(sym)
            e = _token_stop_index(sym)
            if s is None or e is None:
                return CodeLocation()
            return CodeLocation(s, e)

        if not isinstance(obj, ParserRuleContext):
            return CodeLocation()

        start_tok = _ctx_start_token(obj)
        stop_tok = _ctx_stop_token(obj)
        s = _token_start_index(start_tok)
        e = _token_stop_index(stop_tok)
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
    start_tok = _ctx_start_token(ctx)
    stop_tok = _ctx_stop_token(ctx)
    s = _token_start_index(start_tok)
    e = _token_stop_index(stop_tok)
    return _slice_source(code, s, e)


def get_source_text_terminal(node: TerminalNode | None, code: str) -> str:
    if node is None:
        return ""
    sym = node.getSymbol() if hasattr(node, "getSymbol") else getattr(node, "symbol", None)
    s = _token_start_index(sym)
    e = _token_stop_index(sym)
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


def _merge_overlapping_code_locations(blocks: list[CodeBlock]) -> None:
    blocks.sort(key=lambda b: b.loc.firstPos)
    if len(blocks) <= 1:
        return

    merged = [blocks[0]]
    for next_item in blocks[1:]:
        current = merged[-1]
        if next_item.loc.firstPos <= current.loc.lastPos:
            current.loc.lastPos = max(current.loc.lastPos, next_item.loc.lastPos)
        else:
            merged.append(next_item)

    blocks[:] = merged


def _read_file_to_string(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8-sig")


class ELDocument:
    def __init__(self, code: str = "", file_path: Optional[Union[str, Path]] = None):
        self._file_path = Path(file_path) if file_path else Path()
        self._raw_code = ""
        self._parser_objects: Optional[AntlrParserObjects] = None

        self._variable_declarations: dict[str, list[VariableDeclBlock]] = {}
        self._switch_blocks = SwitchBlockContainer()
        self._order_blocks: dict[OrderDirection, list[CodeBlock]] = {}
        self._sl_pt_blocks: dict[SlPtDirective, list[CodeBlock]] = {}
        self._set_exit_on_close_blocks: list[CodeBlock] = []
        self._terminals: list[CodeBlock] = []

        if code:
            self._parser_objects = AntlrParserObjects.from_string(code)
            self._raw_code = code
        else:
            self._raw_code = ""

    @staticmethod
    def from_file(file_path: Union[str, Path]) -> Optional["ELDocument"]:
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            return None
        code = _read_file_to_string(p)
        if not code:
            return None
        return ELDocument(code=code, file_path=p)

    def code(self) -> str:
        return self._raw_code

    def filePath(self) -> str:
        return str(self._file_path) if str(self._file_path) != "." else ""

    def locFrom(self, obj: object) -> CodeLocation:
        return self.loc_from(obj)

    def file_path(self) -> Path:
        return self._file_path

    def loc_from(self, obj: object) -> CodeLocation:
        if not self._parser_objects:
            return CodeLocation()
        return self._parser_objects.loc_from(obj)

    def getCharAt(self, pos: int) -> str:
        return self.get_char_at(pos)

    def getStreamCharAt(self, pos: int) -> str:
        return self.get_stream_char_at(pos)

    def get_char_at(self, pos: int) -> str:
        return self._raw_code[pos]

    def get_stream_char_at(self, pos: int) -> str:
        return self._raw_code[pos]

    def reset(self) -> None:
        self._clear_caches()
        self._parser_objects = AntlrParserObjects.from_string(self._raw_code)

    def variableDeclarations(self, declaration_type: str) -> list[VariableDeclBlock]:
        return self.variable_declarations(declaration_type)

    def allVariableDeclarations(self) -> dict[str, list[VariableDeclBlock]]:
        return self.all_variable_declarations()

    def switchBlocks(self) -> SwitchBlockContainer:
        return self.switch_blocks()

    def orderBlocks(self) -> dict[OrderDirection, list[CodeBlock]]:
        return self.order_blocks()

    def slPtBlocks(self) -> dict[SlPtDirective, list[CodeBlock]]:
        return self.sl_pt_blocks()

    def setExitOnCloseBlocks(self) -> list[CodeBlock]:
        return self.set_exit_on_close_blocks()

    def variableDeclarationsMerged(self, declaration_type: str) -> dict[str, Variable]:
        return self.variable_declarations_merged(declaration_type)

    def isEmpty(self) -> bool:
        return self.is_empty()

    def isValid(self) -> bool:
        return self.is_valid()

    def isValidLocation(self, loc: CodeLocation) -> bool:
        return self.is_valid_location(loc)

    def codeWithLocationsRemoved(self, locations: list[CodeLocation]) -> str:
        return self.code_with_locations_removed(locations)

    def _scoped_code_block(self, ctx: ParserRuleContext) -> CodeBlock:
        enclosing = get_outermost_parent_scope(ctx)
        loc = self.loc_from(enclosing) if enclosing else self.loc_from(ctx)
        code = get_source_text_ctx(enclosing, self._raw_code) if enclosing else get_source_text_ctx(ctx, self._raw_code)
        return CodeBlock(loc, code)

    def variable_declarations(self, declaration_type: str) -> list[VariableDeclBlock]:
        if not self._variable_declarations:
            self._variable_declarations = self._extract_variable_declarations()
        key = _to_lower(declaration_type)
        return self._variable_declarations.get(key, [])

    def all_variable_declarations(self) -> dict[str, list[VariableDeclBlock]]:
        if not self._variable_declarations:
            self._variable_declarations = self._extract_variable_declarations()
        return self._variable_declarations

    def switch_blocks(self) -> SwitchBlockContainer:
        if self._switch_blocks.empty():
            self._switch_blocks = self._extract_switch_blocks()
        return self._switch_blocks

    def order_blocks(self) -> dict[OrderDirection, list[CodeBlock]]:
        if not self._order_blocks:
            self._order_blocks = self._extract_order_blocks()
        return self._order_blocks

    def sl_pt_blocks(self) -> dict[SlPtDirective, list[CodeBlock]]:
        if not self._sl_pt_blocks:
            self._sl_pt_blocks = self._extract_sl_pt_blocks()
        return self._sl_pt_blocks

    def set_exit_on_close_blocks(self) -> list[CodeBlock]:
        if not self._set_exit_on_close_blocks:
            self._set_exit_on_close_blocks = self._extract_set_exit_on_close_blocks()
        return self._set_exit_on_close_blocks

    def variable_declarations_merged(self, declaration_type: str) -> dict[str, Variable]:
        results: dict[str, Variable] = {}
        decl_blocks = self.variable_declarations(declaration_type)
        for block in decl_blocks:
            for var in block:
                if var.name not in results:
                    results[var.name] = var
        return results

    def terminals(self) -> list[CodeBlock]:
        if not self._terminals:
            self._terminals = self._extract_terminals()
        return self._terminals

    def is_empty(self) -> bool:
        return self._raw_code == ""

    def is_valid(self) -> bool:
        return self._parser_objects is not None

    def is_valid_location(self, loc: CodeLocation) -> bool:
        return (
            not self.is_empty()
            and loc.is_valid()
            and loc.firstPos < len(self._raw_code)
            and loc.lastPos < len(self._raw_code)
        )

    def code_with_locations_removed(self, locations: list[CodeLocation]) -> str:
        cleaned = self._raw_code
        sorted_locs = sorted(locations)
        for loc in sorted_locs:
            if not loc.is_valid():
                continue
            cleaned = cleaned[: loc.firstPos] + cleaned[loc.firstPos + loc.length() :]
        return cleaned

    def _parse_tree(self) -> EasyLanguageParser.StartContext:
        if not self._parser_objects:
            raise RuntimeError("Parser objects are not initialized")
        try:
            if hasattr(self._parser_objects.tokens, "seek"):
                self._parser_objects.tokens.seek(0)
        except Exception:
            pass
        try:
            self._parser_objects.parser.reset()
        except Exception:
            pass
        return self._parser_objects.parser.start()

    def _extract_variable_declarations(self) -> dict[str, list[VariableDeclBlock]]:
        if not self._parser_objects:
            return {}
        tree = self._parse_tree()
        results: dict[str, list[VariableDeclBlock]] = {}
        program = tree.program()

        DeclItem = (
            EasyLanguageParser.Input_itemContext
            | EasyLanguageParser.Var_itemContext
            | EasyLanguageParser.Const_itemContext
        )

        def process_declaration(items: list[DeclItem], category: str, block_loc: CodeLocation) -> None:
            block = VariableDeclBlock()
            block.loc = block_loc
            for item in items:
                type_ctx = item.type_ref()
                type_str = get_source_text_ctx(type_ctx, self._raw_code) if type_ctx else ""
                ident_node = item.IDENT()
                name_str = ident_node.getText() if ident_node else ""
                expr_ctx = item.expr()
                value_str = get_source_text_ctx(expr_ctx, self._raw_code) if expr_ctx else ""
                block.add(Variable(type_str, name_str, value_str, self.loc_from(item)))
            results.setdefault(category, []).append(block)

        for decl in program.declaration():
            id_ctx = decl.input_decl()
            if id_ctx:
                items = id_ctx.input_item()
                process_declaration(items, "inputs", self.loc_from(id_ctx))
                continue
            vd_ctx = decl.var_decl()
            if vd_ctx:
                items = vd_ctx.var_item()
                process_declaration(items, "vars", self.loc_from(vd_ctx))
                continue
            cd_ctx = decl.const_decl()
            if cd_ctx:
                items = cd_ctx.const_item()
                process_declaration(items, "consts", self.loc_from(cd_ctx))
                continue

        return results

    def _extract_switch_blocks(self) -> SwitchBlockContainer:
        blocks = SwitchBlockContainer()
        extractor = ELTreeWalker()

        def handler(ctx: EasyLanguageParser.Switch_stmtContext) -> None:
            sb = self._extract_switch_block(ctx)
            blocks.add(sb)

        extractor.register_ctx_handler(EasyLanguageParser.Switch_stmtContext, handler)
        tree = self._parse_tree()
        extractor.walk(tree)
        return blocks

    def _extract_switch_block(self, ctx: EasyLanguageParser.Switch_stmtContext) -> SwitchBlock:
        sb = SwitchBlock()
        sb.loc = self.loc_from(ctx)
        expr_ctx = ctx.expr()
        sb.switchVar = get_source_text_ctx(expr_ctx, self._raw_code) if expr_ctx else ""

        body = ctx.switch_body()
        if body:
            for sec in body.switch_section():
                section_code = ""
                section_loc = CodeLocation()
                stmts = sec.statement()
                if stmts:
                    first = stmts[0]
                    last = stmts[-1]
                    first_start = _token_start_index(_ctx_start_token(first))
                    last_stop = _token_stop_index(_ctx_stop_token(last))
                    if first_start is not None and last_stop is not None:
                        if 0 <= first_start <= last_stop < len(self._raw_code):
                            section_code = self._raw_code[first_start : last_stop + 1]
                    first_loc = self.loc_from(first)
                    last_loc = self.loc_from(last)
                    if first_loc.is_valid() and last_loc.is_valid():
                        section_loc = CodeLocation(first_loc.firstPos, last_loc.lastPos)

                    if len(stmts) == 1:
                        blk = stmts[0].block()
                        if blk:
                            begin_term = blk.BEGIN()
                            end_term = blk.END()
                            begin_sym = begin_term.getSymbol() if begin_term else None
                            end_sym = end_term.getSymbol() if end_term else None
                            begin_stop = _token_stop_index(begin_sym)
                            end_start = _token_start_index(end_sym)
                            if begin_stop is not None and end_start is not None:
                                start_idx = begin_stop + 1
                                end_idx = end_start - 1
                                if 0 <= start_idx <= end_idx < len(self._raw_code):
                                    section_code = self._raw_code[start_idx : end_idx + 1]
                                    section_loc = CodeLocation(start_idx, end_idx)

                                    while (
                                        section_code
                                        and section_code[0] in ("\n", "\r")
                                        and section_loc.firstPos < section_loc.lastPos
                                    ):
                                        section_code = section_code[1:]
                                        section_loc.firstPos += 1
                                    while (
                                        section_code
                                        and section_code[-1] in ("\n", "\r")
                                        and section_loc.firstPos < section_loc.lastPos
                                    ):
                                        section_code = section_code[:-1]
                                        section_loc.lastPos -= 1

                if sec.CASE():
                    labels = sec.case_labels()
                    if labels:
                        for item in labels.case_label_item():
                            cb = CaseBlock(section_loc, get_source_text_ctx(item, self._raw_code), section_code)
                            sb.add(cb)
                elif sec.DEFAULT():
                    cb = CaseBlock(section_loc, "default", section_code)
                    sb.add(cb)
        return sb

    def _extract_order_blocks(self) -> dict[OrderDirection, list[CodeBlock]]:
        results: dict[OrderDirection, list[CodeBlock]] = {}
        extractor = ELTreeWalker()

        def handler(ctx: EasyLanguageParser.Order_stmtContext) -> None:
            cb = self._scoped_code_block(ctx)
            side = ctx.order_side()
            if is_buy_order(side):
                results.setdefault(OrderDirection.LongEntry, []).append(cb)
            elif is_sell_short_order(side):
                results.setdefault(OrderDirection.ShortEntry, []).append(cb)
            elif is_sell_order(side):
                results.setdefault(OrderDirection.LongExit, []).append(cb)
            elif is_buy_to_cover_order(side):
                results.setdefault(OrderDirection.ShortExit, []).append(cb)

        extractor.register_ctx_handler(EasyLanguageParser.Order_stmtContext, handler)
        tree = self._parse_tree()
        extractor.walk(tree)
        for k, vec in results.items():
            _merge_overlapping_code_locations(vec)
        return results

    def _extract_sl_pt_blocks(self) -> dict[SlPtDirective, list[CodeBlock]]:
        results: dict[SlPtDirective, list[CodeBlock]] = {}
        extractor = ELTreeWalker()

        def handler(ctx: EasyLanguageParser.Sl_pt_directive_stmtContext) -> None:
            cb = self._scoped_code_block(ctx)
            if ctx.setStopLossStmt():
                results.setdefault(SlPtDirective.SetStopLoss, []).append(cb)
            elif ctx.setProfitTargetStmt():
                results.setdefault(SlPtDirective.SetProfitTarget, []).append(cb)
            else:
                results.setdefault(SlPtDirective.SetContractOrPosition, []).append(cb)

        extractor.register_ctx_handler(EasyLanguageParser.Sl_pt_directive_stmtContext, handler)
        tree = self._parse_tree()
        extractor.walk(tree)
        for k, vec in results.items():
            _merge_overlapping_code_locations(vec)
        return results

    def _extract_set_exit_on_close_blocks(self) -> list[CodeBlock]:
        results: list[CodeBlock] = []
        extractor = ELTreeWalker()

        def handler(ctx: EasyLanguageParser.SetExitOnCloseStmtContext) -> None:
            results.append(self._scoped_code_block(ctx))

        extractor.register_ctx_handler(EasyLanguageParser.SetExitOnCloseStmtContext, handler)
        tree = self._parse_tree()
        extractor.walk(tree)
        return results

    def _extract_terminals(self) -> list[CodeBlock]:
        results: list[CodeBlock] = []
        extractor = ELTreeWalker()

        def handler(node: TerminalNode) -> None:
            loc = self.loc_from(node)
            code = get_source_text_terminal(node, self._raw_code)
            results.append(CodeBlock(loc, code))

        extractor.register_terminal_handler(handler)
        tree = self._parse_tree()
        extractor.walk(tree)
        return results

    def _clear_caches(self) -> None:
        self._variable_declarations.clear()
        self._switch_blocks.clear()
        self._order_blocks.clear()
        self._sl_pt_blocks.clear()
        self._set_exit_on_close_blocks.clear()


def main():
    el_file_path = Path(__file__).parent / "el_samples" / "ES-106M-B10-R1e-754-1.txt"
    el_code = el_file_path.read_text()
    doc = ELDocument(el_code, el_file_path)

    all_vars = doc.all_variable_declarations()
    for category, decl_blocks in all_vars.items():
        print(f"{category}:")
        for block in decl_blocks:
            for var in block:
                print(
                    f"\t{var.dataType} {var.name}({var.value}) [raw code: `{el_code[var.loc.firstPos:var.loc.lastPos+1]}`]"
                )


if __name__ == "__main__":
    main()
