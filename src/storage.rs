use anyhow::Error;
use std::fs;
use std::io;
use std::io::Read;
use std::io::Seek;
use std::io::Write;
use std::{ops::Range, path::PathBuf};

pub trait Storage {
    type Reader: io::Read + io::Seek;

    fn display(&self) -> String;
    fn reader(&mut self) -> Result<&mut Self::Reader, Error>;
    fn read(&mut self) -> Result<Vec<u8>, Error>;
    fn write(&mut self, bytes: &[u8]) -> Result<(), Error>;
    fn splice(&mut self, range: Range<usize>, bytes: &[u8]) -> Result<(), Error>;
}

pub struct FileStorage {
    path: PathBuf,
    reader: Option<io::BufReader<fs::File>>,
}

impl FileStorage {
    pub fn new(path: PathBuf) -> Self {
        FileStorage { path, reader: None }
    }
}

impl Storage for FileStorage {
    type Reader = io::BufReader<fs::File>;

    fn display(&self) -> String {
        self.path.display().to_string()
    }

    fn reader(&mut self) -> Result<&mut Self::Reader, Error> {
        if self.reader.is_none() {
            self.reader = Some(io::BufReader::new(fs::File::open(&self.path)?));
        }
        Ok(self.reader.as_mut().unwrap())
    }

    fn read(&mut self) -> Result<Vec<u8>, Error> {
        Ok(fs::read(&self.path)?)
    }

    fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.reader = None;
        fs::write(&self.path, bytes)?;
        Ok(())
    }

    fn splice(&mut self, range: Range<usize>, bytes: &[u8]) -> Result<(), Error> {
        // TODO: use fallocate on linux
        self.reader = None;
        let mut file = fs::File::options()
            .read(true)
            .write(true)
            .truncate(false)
            .create(false)
            .open(&self.path)?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)?;
        contents.splice(range, bytes.iter().copied());
        file.set_len(0)?;
        file.seek(io::SeekFrom::Start(0))?;
        file.write_all(&contents)?;
        Ok(())
    }
}
