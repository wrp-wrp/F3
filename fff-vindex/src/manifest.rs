use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseFileBinding {
    pub path: PathBuf,
    pub size: u64,
    pub schema_checksum: u64,
    pub data_checksum: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexKind {
    IvfFlat,
    IvfPq,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    pub name: String,
    pub path: PathBuf,
    pub kind: IndexKind,
    pub vector_leaf_index: u32,
    pub dim: u32,
    pub metric: String,

    #[serde(default)]
    pub build_params: serde_json::Value,
    #[serde(default)]
    pub quantization: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexManifest {
    pub base: BaseFileBinding,
    pub indexes: Vec<IndexEntry>,
}

impl IndexManifest {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).with_context(|| format!("read manifest {}", path.display()))?;
        let manifest: Self =
            serde_json::from_slice(&bytes).with_context(|| "parse manifest json")?;
        Ok(manifest)
    }

    pub fn save_pretty(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create dir {}", parent.display()))?;
        }
        let bytes = serde_json::to_vec_pretty(self).with_context(|| "serialize manifest json")?;
        fs::write(path, bytes).with_context(|| format!("write manifest {}", path.display()))?;
        Ok(())
    }

    pub fn add_or_replace_index(&mut self, entry: IndexEntry) {
        if let Some(existing) = self.indexes.iter_mut().find(|e| e.name == entry.name) {
            *existing = entry;
        } else {
            self.indexes.push(entry);
        }
    }

    pub fn get_by_name(&self, name: &str) -> Result<&IndexEntry> {
        self.indexes
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| anyhow::anyhow!("index not found: {name}"))
    }

    pub fn validate_base(&self) -> Result<()> {
        let meta = fs::metadata(&self.base.path)
            .with_context(|| format!("stat base file {}", self.base.path.display()))?;
        let actual_size = meta.len();
        if actual_size != self.base.size {
            bail!(
                "base file size mismatch: expected {}, got {}",
                self.base.size,
                actual_size
            );
        }
        Ok(())
    }
}

