"use client";

import { useHistory } from "@/lib/store/history";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Trash2, ExternalLink, History as HistoryIcon, Search } from "lucide-react";
import Link from "next/link";
import { Input } from "@/components/ui/input";
import { useState } from "react";
import { getOutputUrl } from "@/lib/api/client";

export default function HistoryPage() {
  const { history, isLoaded, deleteItem, clearHistory } = useHistory();
  const [search, setSearch] = useState("");

  const filteredHistory = history.filter(
    (item) =>
      item.request.question.toLowerCase().includes(search.toLowerCase()) ||
      item.response.answer.toLowerCase().includes(search.toLowerCase())
  );

  if (!isLoaded) {
    return <div className="p-8 text-center text-muted-foreground">Loading history...</div>;
  }

  return (
    <div className="max-w-5xl mx-auto py-8 space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div className="flex items-center gap-3">
          <HistoryIcon className="w-8 h-8 text-primary" />
          <h1 className="text-3xl font-bold">Analysis History</h1>
        </div>
        {history.length > 0 && (
          <Button variant="outline" className="text-destructive hover:bg-destructive/10" onClick={clearHistory}>
            <Trash2 className="w-4 h-4 mr-2" /> Clear All History
          </Button>
        )}
      </div>

      <div className="flex items-center gap-2 max-w-md">
        <Search className="w-5 h-5 text-muted-foreground" />
        <Input 
          placeholder="Search by question or diagnosis..." 
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      {history.length === 0 ? (
        <div className="text-center py-20 border border-dashed rounded-xl bg-muted/20">
          <HistoryIcon className="w-12 h-12 text-muted-foreground mx-auto mb-4 opacity-50" />
          <h3 className="text-lg font-medium">No history found</h3>
          <p className="text-muted-foreground mt-1 mb-6">You havent performed any medical image analysis yet.</p>
          <Link href="/analyze">
            <Button>Start Analysis</Button>
          </Link>
        </div>
      ) : filteredHistory.length === 0 ? (
        <div className="text-center py-20">
          <p className="text-muted-foreground">No results match your search.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredHistory.map((item) => (
            <Card key={item.id} className="overflow-hidden flex flex-col hover:shadow-md transition-shadow">
              <div className="h-48 bg-black/5 flex items-center justify-center relative overflow-hidden">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img 
                  src={item.originalImageBase64} 
                  alt="Medical scan" 
                  className="w-full h-full object-cover opacity-80"
                />
                {item.response.heatmap_url && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img 
                    src={getOutputUrl(item.response.heatmap_url)} 
                    alt="Heatmap" 
                    className="absolute inset-0 w-full h-full object-cover mix-blend-overlay opacity-70"
                  />
                )}
              </div>
              <CardHeader className="pb-3 flex-1">
                <CardDescription className="text-xs">{new Date(item.timestamp).toLocaleString()}</CardDescription>
                <CardTitle className="text-base leading-snug line-clamp-2 mt-1">
                  Q: {item.request.question}
                </CardTitle>
                <p className="text-sm text-muted-foreground line-clamp-3 mt-2 font-medium">
                  A: {item.response.answer}
                </p>
              </CardHeader>
              <CardContent className="pt-0 flex justify-between items-center border-t p-4 mt-auto">
                <Button variant="ghost" size="sm" className="text-destructive hover:bg-destructive/10 hover:text-destructive" onClick={() => deleteItem(item.id)}>
                  <Trash2 className="w-4 h-4" />
                </Button>
                <Link href={`/heatmap/${item.id}`}>
                  <Button size="sm" variant="secondary" className="gap-2">
                    View Details <ExternalLink className="w-4 h-4" />
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
