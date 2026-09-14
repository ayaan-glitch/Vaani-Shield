import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import '../core/theme/app_theme.dart';
import '../models/detection_result.dart';

// ─────────────────────────────────────────────────────────────
// ProbabilityGraph – scrolling line chart
// INSUFFICIENT_AUDIO points shown as gaps (null spots).
// Driven by a stream; caller passes the data list.
// ─────────────────────────────────────────────────────────────
class ProbabilityGraph extends StatelessWidget {
  /// List of results in chronological order (max ~60 entries).
  final List<DetectionResult> history;

  const ProbabilityGraph({super.key, required this.history});

  @override
  Widget build(BuildContext context) {
    final spots = <FlSpot>[];
    for (var i = 0; i < history.length; i++) {
      final r = history[i];
      // Skip INSUFFICIENT_AUDIO – treated as a gap
      if (r.state == DetectionState.INSUFFICIENT_AUDIO) continue;
      spots.add(FlSpot(i.toDouble(), r.probability * 100));
    }

    if (spots.isEmpty) {
      return _emptyGraph();
    }

    return LineChart(
      LineChartData(
        minY: 0,
        maxY: 100,
        clipData: const FlClipData.all(),
        gridData: FlGridData(
          show: true,
          horizontalInterval: 25,
          getDrawingHorizontalLine: (_) => const FlLine(
            color: AppColors.border,
            strokeWidth: 1,
          ),
          drawVerticalLine: false,
        ),
        borderData: FlBorderData(show: false),
        titlesData: FlTitlesData(
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              interval: 25,
              reservedSize: 30,
              getTitlesWidget: (value, _) => Text(
                '${value.toInt()}%',
                style: const TextStyle(
                    color: AppColors.textSecondary, fontSize: 9),
              ),
            ),
          ),
          bottomTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
        ),
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: true,
            color: AppColors.accent,
            barWidth: 2,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(
              show: true,
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  AppColors.accent.withOpacity(0.3),
                  AppColors.accent.withOpacity(0.0),
                ],
              ),
            ),
          ),
        ],
        // Threshold lines
        extraLinesData: ExtraLinesData(horizontalLines: [
          HorizontalLine(
            y: 70,
            color: AppColors.highRisk.withOpacity(0.5),
            strokeWidth: 1,
            dashArray: [4, 4],
            label: HorizontalLineLabel(
              show: true,
              alignment: Alignment.topRight,
              labelResolver: (_) => 'HIGH RISK',
              style: const TextStyle(
                  color: AppColors.highRisk, fontSize: 8, letterSpacing: 0.5),
            ),
          ),
          HorizontalLine(
            y: 40,
            color: AppColors.suspicious.withOpacity(0.5),
            strokeWidth: 1,
            dashArray: [4, 4],
            label: HorizontalLineLabel(
              show: true,
              alignment: Alignment.topRight,
              labelResolver: (_) => 'SUSPICIOUS',
              style: const TextStyle(
                  color: AppColors.suspicious, fontSize: 8, letterSpacing: 0.5),
            ),
          ),
        ]),
      ),
      duration: const Duration(milliseconds: 200),
    );
  }

  Widget _emptyGraph() {
    return const Center(
      child: Text(
        'Waiting for audio…',
        style: TextStyle(color: AppColors.textSecondary, fontSize: 12),
      ),
    );
  }
}
