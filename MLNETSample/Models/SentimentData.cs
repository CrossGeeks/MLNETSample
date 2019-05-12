using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNETSample.Models
{
    class SentimentData
    {
        [LoadColumn(0)]
        public bool Label { get; set; }

        [LoadColumn(1)]
        public string Text { get; set; }
    }
}
