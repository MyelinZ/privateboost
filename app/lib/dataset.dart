import 'dart:convert';

import 'package:flutter/services.dart' show rootBundle;

/// One training record: a dataset's feature values in file-column order,
/// paired with the binary `target` label.
typedef TrainRow = (List<double> features, double label);

/// Bundled train split per dataset id: the source CSV's first
/// `(rowCount * 0.8).floor()` rows, the same split `pbr-e2e` applies to compute
/// the held-out remainder, so no bundled row is one the aggregator scores
/// against.
const Map<String, String> _trainAssetPaths = {
  'heart_disease': 'assets/heart_disease_train.csv',
  'breast_cancer': 'assets/breast_cancer_train.csv',
  'pima_diabetes': 'assets/pima_diabetes_train.csv',
  'cdc_diabetes': 'assets/cdc_diabetes_train.csv',
};

final Map<String, List<TrainRow>> _cache = {};

/// Each data row's features, every column except `target` in file order, with
/// its label. Cached per dataset after the first load.
///
/// Throws [ArgumentError] if [datasetId] names no bundled dataset: a device
/// asked for one it does not have must say so, not return an empty slice.
Future<List<TrainRow>> loadTrainRows([String datasetId = 'heart_disease']) async {
  final cached = _cache[datasetId];
  if (cached != null) return cached;
  final path = _trainAssetPaths[datasetId];
  if (path == null) {
    throw ArgumentError.value(
      datasetId,
      'datasetId',
      'no bundled dataset; must be one of ${_trainAssetPaths.keys.join(', ')}',
    );
  }
  final raw = await rootBundle.loadString(path);
  final lines =
      const LineSplitter().convert(raw).where((l) => l.trim().isNotEmpty).toList();
  final header = lines.first.trim().split(',');
  final labelIdx = header.indexOf('target');
  final rows = <TrainRow>[];
  for (final line in lines.skip(1)) {
    final cells = line.trim().split(',');
    final features = <double>[];
    for (var i = 0; i < cells.length; i++) {
      if (i == labelIdx) continue;
      features.add(double.parse(cells[i]));
    }
    rows.add((features, double.parse(cells[labelIdx])));
  }
  return _cache[datasetId] = rows;
}

/// The feature width of [datasetId]'s bundled TRAIN split, read from the
/// parsed data itself rather than a hardcoded table, so the reported width
/// can never drift from what [loadTrainRows] actually parses.
Future<int> datasetFeatureCount(String datasetId) async {
  final rows = await loadTrainRows(datasetId);
  return rows.first.$1.length;
}

/// This device's contiguous slice of [rows] under a fleet split into
/// [batchCount] equal parts, selecting part [batchId] (0-indexed). The slice is
/// `[batchId*L~/batchCount .. (batchId+1)*L~/batchCount)`, with the last part
/// running to the end so the [batchCount] slices tile all `L` rows with no gap
/// or overlap. Returns the slice rows plus its `[start, end)` row range.
///
/// Throws [ArgumentError] if [batchId] is not in `[0, batchCount)` or the slice
/// would be empty, since an empty batch would enroll no clients.
({List<TrainRow> rows, int start, int end}) batchSlice(
  List<TrainRow> rows,
  int batchId,
  int batchCount,
) {
  if (batchId < 0 || batchId >= batchCount) {
    throw ArgumentError.value(
      batchId,
      'batchId',
      'must be in [0, $batchCount)',
    );
  }
  final l = rows.length;
  final start = batchId * l ~/ batchCount;
  final end = batchId == batchCount - 1 ? l : (batchId + 1) * l ~/ batchCount;
  if (start >= end) {
    throw ArgumentError(
      'batch $batchId of $batchCount over $l rows is empty',
    );
  }
  return (rows: rows.sublist(start, end), start: start, end: end);
}
