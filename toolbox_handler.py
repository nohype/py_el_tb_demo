import datetime as dt
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from pprint import pprint
from typing import Any, Literal, NamedTuple, cast, get_args

import pandas as pd
import xlwings as xw
import xlwings.utils as xwu


def _col_index(col_name: str) -> int:
    return xwu.column_to_number(col_name) - 1


def _row_index(row_number: int) -> int:
    return row_number - 1


class CellIndex(NamedTuple):
    row: int
    col: int

    @staticmethod
    def from_address(addr: str):
        index_tuple = xwu.address_to_index_tuple(addr)
        if index_tuple is None:
            raise ValueError("Failed to convert string '{addr}' to Excel index tuple.")

        row_1based, col_1based = index_tuple
        return CellIndex(row_1based - 1, col_1based - 1)


class TBSettings:
    is_report_dir: Path
    oos_report_dir: Path
    candidate_gen_dir: Path
    candidate_code_dir: Path
    ts_wfo_reports_dir: Path
    history_code_dir: Path


RobustnessLevel = Literal["RL1", "RL2", "RL2 ADV", "RL3", "RL3 ADV"]
__ROBUSTNESS_LEVELS = get_args(RobustnessLevel)


def get_robustness_levels_from(level: RobustnessLevel):
    index = __ROBUSTNESS_LEVELS.index(level)
    levels_after = __ROBUSTNESS_LEVELS[index:]
    return list(levels_after)


DB_NAME_COL = _col_index("A")
DB_PREVAL_RESULT_COL = _col_index("R")
DB_VALIDATION_RESULT_COL = _col_index("AD")
DB_OPT_INPUTS_COL = _col_index("AO")
DB_FIRST_DATA_ROW = _row_index(4)

SEL_SELECTION_COL = _col_index("A")
SEL_NAME_COL = _col_index("B")
SEL_SWITCH_COLS = [_col_index(col_name) for col_name in ["C", "D", "E"]]
SEL_FIRST_DATA_ROW = _row_index(13)
SEL_MARKET_CELL = CellIndex.from_address("L4")
SEL_TIMEFRAME_CELL = CellIndex.from_address("M4")
SEL_RUN_CELL = CellIndex.from_address("N4")

DNP_FIRST_DATA_ROW = _row_index(7)
DNP_NAME_COL = _col_index("A")
DNP_VALUE_COL = _col_index("B")
DNP_MODE_COL = _col_index("C")

SETTING_IS_DIR_ROW = _row_index(6)
SETTING_IS_DIR_COL = _col_index("I")
SETTING_OOS_DIR_ROW = _row_index(9)
SETTING_OOS_DIR_COL = _col_index("I")
SETTING_CANDIDATE_GEN_DIR_ROW = _row_index(12)
SETTING_CANDIDATE_GEN_DIR_COL = _col_index("I")
SETTING_STRAT_CODE_DIR_ROW = _row_index(15)
SETTING_STRAT_CODE_DIR_COL = _col_index("I")
SETTING_TS_WFO_REPORTS_DIR_ROW = _row_index(18)
SETTING_TS_WFO_REPORTS_DIR_COL = _col_index("I")
SETTING_HISTORY_CODE_DIR_ROW = _row_index(21)
SETTING_HISTORY_CODE_DIR_COL = _col_index("I")
SETTING_FIRST_DATA_ROW = _row_index(6)
SETTING_TEMPLATES_COL = _col_index("Q")
SETTING_START_TIME_COL = _col_index("R")
SETTING_END_TIME_COL = _col_index("S")

PREPARE_MKT_CELL = CellIndex.from_address("A4")
PREPARE_LSB_CELL = CellIndex.from_address("B4")
PREPARE_TIMEFRAME_CELL = CellIndex.from_address("C4")
PREPARE_SESSION_CELL = CellIndex.from_address("D4")
PREPARE_BLOCK_CELL = CellIndex.from_address("E4")
PREPARE_DTSWING_CELL = CellIndex.from_address("F4")
PREPARE_BLOCK_COUNT_CELL = CellIndex.from_address("G4")
PREPARE_MQ_CELL = CellIndex.from_address("H4")
PREPARE_LAST_DATE_CELL = CellIndex.from_address("B8")
PREPARE_DAILY_USED_CELL = CellIndex.from_address("B9")
PREPARE_SESS_START_CELL = CellIndex.from_address("D10")
PREPARE_SESS_END_CELL = CellIndex.from_address("F10")
PREPARE_MAX_BARS_BACK_CELL = CellIndex.from_address("B11")


_strip_and_casefold = lambda x: x.strip().casefold()


def string_to_time(s: str) -> dt.time:
    return dt.datetime.strptime(s, "%H:%M").time()


def float_to_time(f: float) -> dt.time:
    seconds = round(f * 24 * 3600)
    return dt.time((seconds // 3600) % 24, (seconds % 3600) // 60, seconds % 60)


def time_to_float(input_time: dt.datetime | str) -> float:
    t = string_to_time(input_time) if isinstance(input_time, str) else input_time
    return ((t.hour * 3600) + (t.minute * 60) + t.second) / (24 * 3600)


class ToolboxHandler:
    def __init__(self, workbook_path: str, close_new_instance: bool = True):
        self.__wb_path = workbook_path
        self.__was_already_open = not close_new_instance

        wb: xw.Book | None = None
        for app in xw.apps:
            for book in app.books:
                if Path(book.fullname).samefile(self.__wb_path):
                    self.__was_already_open = True
                    wb = book
                    break

        if wb is None:
            wb = xw.Book(self.__wb_path)

        if wb is None:
            raise RuntimeError(f"Failed to open '{self.__wb_path}' in Excel.")

        self.__wb: xw.Book = wb

        self.__setting_sheet = None
        self.__prepare_sheet = None
        self.__dnp_sheet = None
        self.__select_sheet = None
        self.__pre_validate_sheet = None
        self.__validate_sheet = None
        self.__complete_sheet = None
        self.__database_sheet = None
        self.__is_sheet = None
        self.__oos_sheet = None

    def __del__(self):
        if not self.__was_already_open:
            self.__wb.save()
            self.__wb.close()

    def setting_sheet(self) -> xw.Sheet:
        if not self.__setting_sheet:
            self.__setting_sheet = self.__wb.sheets["SETTING"]

        return self.__setting_sheet

    def prepare_sheet(self) -> xw.Sheet:
        if not self.__prepare_sheet:
            self.__prepare_sheet = self.__wb.sheets["1. PREPARE"]

        return self.__prepare_sheet

    def select_sheet(self) -> xw.Sheet:
        if not self.__select_sheet:
            self.__select_sheet = self.__wb.sheets["3. SELECT"]

        return self.__select_sheet

    def database_sheet(self) -> xw.Sheet:
        if not self.__database_sheet:
            self.__database_sheet = self.__wb.sheets["DATABASE"]

        return self.__database_sheet

    def dnp_sheet(self) -> xw.Sheet:
        if not self.__dnp_sheet:
            self.__dnp_sheet = self.__wb.sheets["2. D&P"]

        return self.__dnp_sheet

    def is_sheet(self) -> xw.Sheet:
        if not self.__is_sheet:
            self.__is_sheet = self.__wb.sheets["IS"]

        return self.__is_sheet

    def get_settings(self) -> TBSettings:
        ss = self.setting_sheet()

        settings = TBSettings()
        settings.is_report_dir = Path(cast(str, ss.cells[(SETTING_IS_DIR_ROW, SETTING_IS_DIR_COL)].value))
        settings.oos_report_dir = Path(cast(str, ss.cells[(SETTING_OOS_DIR_ROW, SETTING_OOS_DIR_COL)].value))
        settings.candidate_gen_dir = Path(
            cast(str, ss.cells[(SETTING_CANDIDATE_GEN_DIR_ROW, SETTING_CANDIDATE_GEN_DIR_COL)].value)
        )
        settings.candidate_code_dir = Path(
            cast(str, ss.cells[(SETTING_STRAT_CODE_DIR_ROW, SETTING_STRAT_CODE_DIR_COL)].value)
        )
        settings.ts_wfo_reports_dir = Path(
            cast(str, ss.cells[(SETTING_TS_WFO_REPORTS_DIR_ROW, SETTING_TS_WFO_REPORTS_DIR_COL)].value)
        )
        settings.history_code_dir = Path(
            cast(str, ss.cells[(SETTING_HISTORY_CODE_DIR_ROW, SETTING_HISTORY_CODE_DIR_COL)].value)
        )

        return settings

    def get_database_column(
        self,
        col_index: int,
        default_value: Any = None,
        name_filter_patterns: list[str] = [],
        result_filters: list[str] = [],
    ) -> dict[str, str]:
        results = {}
        db = self.database_sheet()
        last_data_row = db.cells[DB_FIRST_DATA_ROW:, DB_NAME_COL].last_cell.end("up").row - 1

        print(
            f"getting data range for db.cells[{DB_FIRST_DATA_ROW} : {last_data_row + 1}, {DB_NAME_COL} : {col_index + 1}]"
        )
        data_range = db.cells[DB_FIRST_DATA_ROW : last_data_row + 1, DB_NAME_COL : col_index + 1]

        # load sheet data into a DataFrame
        df = pd.DataFrame(data_range.value)

        # filter rows based on names matching name_filter_patterns
        if name_filter_patterns:
            filtered_df = df[
                df.iloc[:, 0].apply(
                    lambda x: (
                        any(fnmatch(x, pattern) for pattern in name_filter_patterns) if isinstance(x, str) else False
                    )
                )
            ]
            filtered_df = cast(pd.DataFrame, filtered_df)
        else:
            filtered_df = df

        # replace `None`/na values in the target colum by the default value
        filtered_df.iloc[:, col_index] = filtered_df.iloc[:, col_index].fillna(default_value)

        # filter rows based on the target column value matching result_filters
        if result_filters:
            filtered_df = filtered_df[
                filtered_df.iloc[:, col_index].apply(
                    lambda x: (any(fnmatch(x, pattern) for pattern in result_filters) if isinstance(x, str) else False)
                )
            ]

        # construct a {name: target_value} dict
        results = dict(zip(filtered_df.iloc[:, 0], filtered_df.iloc[:, col_index]))
        return results

    def get_preval_results(
        self, name_filter_patterns: list[str] = [], result_filters: list[str] = []
    ) -> dict[str, str]:
        return self.get_database_column(DB_PREVAL_RESULT_COL, "<no_result>", name_filter_patterns, result_filters)

    def get_passed_preval_strategies(self, name_filter_patterns: list[str] = []) -> list[str]:
        result_dict = self.get_preval_results(name_filter_patterns=name_filter_patterns, result_filters=["PASS*"])
        return list(result_dict.keys())

    def get_failed_preval_strategies(self, name_filter_patterns: list[str] = []) -> list[str]:
        result_dict = self.get_preval_results(name_filter_patterns=name_filter_patterns, result_filters=["FAIL*"])
        return list(result_dict.keys())

    def get_validation_candidates(self, name_filter_patterns: list[str] = []) -> list[str]:
        preval_passed_strats = self.get_passed_preval_strategies(name_filter_patterns)
        if not preval_passed_strats:
            return []

        preval_passed_no_val_result_strats = self.get_database_column(
            DB_VALIDATION_RESULT_COL,
            default_value="<no_result>",
            name_filter_patterns=preval_passed_strats,
            result_filters=["<no_result>"],
        )
        return list(preval_passed_no_val_result_strats.keys())

    def get_validation_results(
        self, name_filter_patterns: list[str] = [], result_filters: list[str] = []
    ) -> dict[str, str]:
        return self.get_database_column(
            DB_VALIDATION_RESULT_COL,
            "<no_result>",
            name_filter_patterns,
            result_filters,
        )

    def get_passed_validation_strategies(
        self,
        name_filter_patterns: list[str] = [],
        min_rl: RobustnessLevel = "RL1",
    ) -> list[str]:
        allowed_rls = get_robustness_levels_from(min_rl)
        result_dict = self.get_validation_results(
            name_filter_patterns=name_filter_patterns, result_filters=[f"PASS* {rl}" for rl in allowed_rls]
        )
        return list(result_dict.keys())

    def get_failed_validation_strategies(self, name_filter_patterns: list[str] = []) -> list[str]:
        result_dict = self.get_validation_results(name_filter_patterns=name_filter_patterns, result_filters=["FAIL*"])
        return list(result_dict.keys())

    def get_strategy_opt_inputs(self, name_filter_patterns: list[str] = []) -> dict[str, list[str]]:
        result_dict = self.get_database_column(DB_OPT_INPUTS_COL, "", name_filter_patterns)

        def split_opt_inputs(opt_inputs: str) -> list[str]:
            return [inp for inp in opt_inputs.split(",") if len(inp.strip()) > 0]

        return {name: split_opt_inputs(opt_inputs) for name, opt_inputs in result_dict.items()}

    def get_strategies_without_opt_inputs(self, name_filter_patterns: list[str] = []) -> list[str]:
        result_dict = self.get_strategy_opt_inputs(name_filter_patterns)
        return [name for name, opt_inputs in result_dict.items() if len(opt_inputs) < 1]

    def set_strategy_opt_inputs(self, strategy_opt_inputs: dict[str, list[str]]):
        db = self.database_sheet()
        last_data_row = db.cells[DB_FIRST_DATA_ROW:, DB_NAME_COL].last_cell.end("up").row - 1

        print(f"getting data range for db.cells[{DB_FIRST_DATA_ROW} : {last_data_row + 1}, {DB_NAME_COL}]")
        name_range = db.cells[DB_FIRST_DATA_ROW : last_data_row + 1, DB_NAME_COL]
        assert name_range.value is not None, "Bad: range of DB strategy names is None!"

        for index, name_value in enumerate(name_range.value):
            name = str(name_value)
            if name not in strategy_opt_inputs:
                continue

            db.cells[(index, DB_OPT_INPUTS_COL)].value = ",".join(strategy_opt_inputs[name])

    def set_popups_enabled(self, enable: bool):
        self.__wb.macro("PTB_SetPopupsEnabled")(enable)
