import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:data_table_2/data_table_2.dart';
import 'dart:convert';

/// A widget that fetches, parses, and displays a CSV file in a responsive table.
class CsvDataTableWidget extends StatefulWidget {
  final String csvUrl;
  final String title;

  const CsvDataTableWidget({
    super.key,
    required this.csvUrl,
    required this.title,
  });

  @override
  State<CsvDataTableWidget> createState() => _CsvDataTableWidgetState();
}

class _CsvDataTableWidgetState extends State<CsvDataTableWidget> {
  bool _isLoading = true;
  String? _error;
  List<String> _headers = [];
  List<List<String>> _rows = [];

  @override
  void initState() {
    super.initState();
    _fetchAndParseCsv();
  }

  @override
  void didUpdateWidget(CsvDataTableWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.csvUrl != widget.csvUrl) {
      _fetchAndParseCsv();
    }
  }

  Future<void> _fetchAndParseCsv() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final response = await http.get(Uri.parse(widget.csvUrl)).timeout(const Duration(seconds: 30));
      
      if (response.statusCode != 200) {
        throw Exception('Failed to load CSV (${response.statusCode})');
      }

      final csvText = utf8.decode(response.bodyBytes);
      final lines = csvText.trim().split('\n');
      
      if (lines.isEmpty) {
        throw Exception('CSV file is empty');
      }

      // Parse headers
      final headerLine = lines[0];
      _headers = _splitCsvLine(headerLine);

      // Parse rows
      _rows = [];
      for (int i = 1; i < lines.length; i++) {
        final rowLine = lines[i];
        if (rowLine.trim().isEmpty) continue;
        _rows.add(_splitCsvLine(rowLine));
      }

      if (mounted) setState(() => _isLoading = false);
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _error = e.toString();
        });
      }
    }
  }

  List<String> _splitCsvLine(String line) {
    final result = <String>[];
    final buffer = StringBuffer();
    bool inQuotes = false;
    
    for (int i = 0; i < line.length; i++) {
      final ch = line[i];
      if (ch == '"') {
        inQuotes = !inQuotes;
      } else if (ch == ',' && !inQuotes) {
        result.add(buffer.toString().trim());
        buffer.clear();
      } else {
        buffer.write(ch);
      }
    }
    result.add(buffer.toString().trim());
    return result;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    if (_isLoading) {
      return Container(
        height: 300,
        decoration: BoxDecoration(
          color: colorScheme.surfaceContainerHighest.withOpacity(0.3),
          borderRadius: BorderRadius.circular(12),
        ),
        child: const Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 16),
              Text('Downloading report...'),
            ],
          ),
        ),
      );
    }

    if (_error != null) {
      return Container(
        height: 200,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: colorScheme.errorContainer.withOpacity(0.2),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: colorScheme.error.withOpacity(0.1)),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.error_outline, color: colorScheme.error, size: 32),
              const SizedBox(height: 12),
              Text(
                'Error loading report',
                style: TextStyle(color: colorScheme.error, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              Text(
                _error!,
                textAlign: TextAlign.center,
                style: TextStyle(color: colorScheme.onErrorContainer, fontSize: 12),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
              const SizedBox(height: 12),
              TextButton(
                onPressed: _fetchAndParseCsv,
                child: const Text('Try Again'),
              ),
            ],
          ),
        ),
      );
    }

    return Container(
      height: 450, // Fixed height for the table card
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: colorScheme.outline.withOpacity(0.1)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.02),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              children: [
                Icon(Icons.table_chart_outlined, size: 18, color: colorScheme.primary),
                const SizedBox(width: 8),
                Text(
                  widget.title,
                  style: theme.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.bold),
                ),
                const Spacer(),
                Text(
                  '${_rows.length} rows',
                  style: theme.textTheme.labelSmall?.copyWith(color: colorScheme.onSurfaceVariant),
                ),
              ],
            ),
          ),
          const Divider(height: 1),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 4.0),
              child: DataTable2(
                columnSpacing: 12,
                horizontalMargin: 12,
                minWidth: 800,
                headingRowHeight: 40,
                dataRowHeight: 48,
                headingTextStyle: theme.textTheme.labelMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: colorScheme.onSurface,
                ),
                columns: _headers.map((h) => DataColumn2(
                  label: Text(h.replaceAll('_', ' ').toUpperCase()),
                  size: ColumnSize.L,
                )).toList(),
                rows: _rows.map((row) => DataRow(
                  cells: row.map((cell) => DataCell(
                    Text(
                      cell,
                      style: theme.textTheme.bodySmall,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  )).toList(),
                )).toList(),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
