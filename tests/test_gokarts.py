import unittest
from pandas.testing import assert_frame_equal
import scripts.gokarts_race as kart_race
import pandas as pd


class TestGoKartRace(unittest.TestCase):
    def setUp(self) -> None:
        self.race = kart_race.Race()
        return super().setUp()

    def test_get_race_results(self):
        race_results = self.race.get_race_results()
        self.assertEqual(len(race_results), 160)

    def test_get_race_finish_times(self):
        race_finish_times = self.race.get_race_finish_times()
        self.assertEqual(race_finish_times["Jeremy"].loc[1], 984.946)

    def test_get_lap_count(self):
        lap_count = self.race.get_lap_count()
        self.assertEqual(lap_count["Pat"].loc[1], 36)
        self.assertEqual(lap_count["Jeremy"].loc[4], 43)

    def test_get_average_lap(self):
        average_lap = self.race.get_mean_lap()
        self.assertAlmostEqual(average_lap["Pat"].loc[1], 27.448833333333337)
        self.assertAlmostEqual(average_lap["Jeremy"].loc[4], 22.616860465116282)

    def test_get_race_ranking(self):
        rankings = self.race.get_race_ranking()
        self.assertEqual(rankings["Jeremy"].loc[3], 1)
        self.assertEqual(rankings["Jeremy"].loc[4], 1)

    def test_get_ranking_summary_table(self):
        ranking_summary = self.race.get_race_ranking_summary_table()
        self.assertEqual(ranking_summary["Mario Kart Points"].iloc[0], 46)

    def test_get_ranking_summary_table(self):
        ranking_summary = self.race.get_race_ranking_summary_table()
        self.assertEqual(ranking_summary["Mario Kart Points"].iloc[0], 46)

    def test_get_fastest_lap(self):
        fastest_laps = self.race.get_fastest_lap()
        self.assertEqual(fastest_laps["Jeremy"].loc[1], 21.221)
        self.assertEqual(fastest_laps["Jeremy"].loc[4], 19.73)

    def test_fastest_lap_ranking(self):
        rankings = self.race.get_fastest_lap_ranking()
        self.assertEqual(rankings["Jeremy"].loc[3], 1)
        self.assertEqual(rankings["Jeremy"].loc[4], 2)

    def test_get_filtered_lap_mean(self):
        filtered_lap_mean = self.race.get_filtered_lap_mean()
        self.assertAlmostEqual(filtered_lap_mean["Jeremy"].loc[3], 21.540970588235297)
        self.assertAlmostEqual(filtered_lap_mean["Jeremy"].loc[4], 20.415950000000002)

    def test_filtered_lap_ranking(self):
        rankings = self.race.get_fastest_lap_ranking()
        self.assertEqual(rankings["Jeremy"].loc[3], 1)
        self.assertEqual(rankings["Jeremy"].loc[4], 2)

    def test_get_kart_fastest_laps(self):
        kart_fastest_lap = self.race.get_kart_fastest_lap()
        self.assertEqual(kart_fastest_lap.loc[1].item(), 19.657)
        self.assertEqual(kart_fastest_lap.loc[21].item(), 19.754)

    def test_flatten(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        self.assertListEqual(self.race.flatten(df).tolist(), [1, 4, 2, 5, 3, 6])

    def test_get_start_positions(self):
        start_positions = self.race.get_start_positions()
        self.assertEqual(start_positions["Jeremy"].loc[3].item(), 1)
        self.assertEqual(start_positions["Pat"].loc[1].item(), 1)

    def test_make_results_table(self):
        df = self.race.make_results_table()
        self.assertEqual(
            df["Driver"][(df["Race Number"] == 1) & (df["Kart Number"] == 10)].item(),
            "Jeremy",
        )
        self.assertEqual(df.size, 196)


class TestGoKartRegression(unittest.TestCase):
    def setUp(self) -> None:
        df = kart_race.Race().make_results_table()
        self.regression = kart_race.Regression(df)
        return super().setUp()

    def test_one_hot_encode(self):
        self.assertEqual(self.regression.one_hot_encode("Driver").size, 336)


class TestGoKartPoints(unittest.TestCase):
    def setUp(self) -> None:
        self.points = kart_race.Points()
        return super().setUp()

    def test_mario_kart(self):
        self.assertEqual(self.points.mario_kart()[1], 15)

    def test_f1_position(self):
        self.assertEqual(self.points.f1_position()[1], 25)

    def test_f1_sprint(self):
        self.assertEqual(self.points.f1_sprint()[1], 8)
