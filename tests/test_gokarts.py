import unittest
from pandas.util.testing import assert_frame_equal
import scripts.gokarts_race as kart_race


class TestGoKartRace(unittest.TestCase):
    def setUp(self) -> None:
        self.race = kart_race.Race()
        return super().setUp()

    def test_get_race_results(self):
        race_results = self.race.get_race_results()
        self.assertEqual(len(race_results), 160)

    # def test_get_race_finish_times(self):
    #     race = kart_race.Race()
    #     race.get_race_finish_times()

    def test_get_lap_count(self):
        lap_count = self.race.get_lap_count()
        self.assertEqual(lap_count["Pat"].loc[1], 36)
        self.assertEqual(lap_count["Jeremy"].loc[4], 43)

    def test_get_average_lap(self):
        average_lap = self.race.get_average_lap()
        self.assertAlmostEqual(average_lap["Pat"].loc[1], 27.448833333333337)
        self.assertAlmostEqual(average_lap["Jeremy"].loc[4], 22.616860465116282)

    def test_get_race_ranking(self):
        rankings = self.race.get_race_ranking()
        self.assertEqual(rankings["Jeremy"].loc[3], 1)
        self.assertEqual(rankings["Jeremy"].loc[4], 1)


class TestGoKartPoints(unittest.TestCase):
    def setUp(self) -> None:
        self.points = kart_race.Points()
        return super().setUp()

    def test_mario_kart_points(self):
        self.assertEqual(self.points.mario_kart_points()[1], 15)

    def test_f1_position_points(self):
        self.assertEqual(self.points.f1_position_points()[1], 25)

    def test_f1_sprint_points(self):
        self.assertEqual(self.points.f1_sprint_points()[1], 8)
