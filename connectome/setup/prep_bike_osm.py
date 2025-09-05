
import osmium
import os
import shutil

#preprocess OSM data to load into r5 for access measurement
#specifically, add LTS values based on bike lanes
#adds LTS 1 to physically protected lanes
#adds LTS 2 painted lanes on residential and tertiary streets
#adds LTS 3 to painted lanes on primary and secondary streets
#adds LTS 4 to everything else

class SimplestLTSAdder(osmium.SimpleHandler):
	def __init__(self, writer, max_lts):
		osmium.SimpleHandler.__init__(self)
		self.writer = writer
		self.n_modified_ways = 0
		self.max_lts = max_lts #if 0, set all
		
	def node(self, n):
		self.writer.add_node(n)
	def way(self, way):
		if 'highway' in way.tags:
			if self.max_lts == 0:
				newtags = dict(way.tags)
				newtags['lts'] = '4'
				self.writer.add_way(way.replace(tags=newtags))
			else:
				if way.tags.get('highway') == 'cycleway': #LTS 1
					newtags = dict(way.tags)
					lts_to_assign = 1
					writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
					newtags['lts'] = str(writeval)
					self.writer.add_way(way.replace(tags=newtags))
					if writeval != 4: self.n_modified_ways += 1
				elif way.tags.get('cycleway') == 'track': #LTS 1
					newtags = dict(way.tags)
					lts_to_assign = 1
					writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
					newtags['lts'] = str(writeval)
					self.writer.add_way(way.replace(tags=newtags))
					if writeval != 4: self.n_modified_ways += 1
				elif way.tags.get('cycleway:left') == 'track': #LTS 1
					newtags = dict(way.tags)
					lts_to_assign = 1
					writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
					newtags['lts'] = str(writeval)
					self.writer.add_way(way.replace(tags=newtags))
					if writeval != 4: self.n_modified_ways += 1
				elif way.tags.get('cycleway:right') == 'track': #LTS 1
					newtags = dict(way.tags)
					lts_to_assign = 1
					writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
					newtags['lts'] = str(writeval)
					self.writer.add_way(way.replace(tags=newtags))
					if writeval != 4: self.n_modified_ways += 1
				elif way.tags.get('cycleway') == 'lane' and way.tags.get('highway') in ['tertiary', 'residential']: #LTS 2
					newtags = dict(way.tags)
					lts_to_assign = 2
					writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
					newtags['lts'] = str(writeval)
					self.writer.add_way(way.replace(tags=newtags))
					if writeval != 4: self.n_modified_ways += 1
				elif way.tags.get('cycleway:left') == 'lane' and way.tags.get('highway') in ['tertiary', 'residential']:#LTS 2
					newtags = dict(way.tags)
					lts_to_assign = 2
					writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
					newtags['lts'] = str(writeval)
					self.writer.add_way(way.replace(tags=newtags))
					if writeval != 4: self.n_modified_ways += 1
				elif way.tags.get('cycleway:right') == 'lane' and way.tags.get('highway') in ['tertiary', 'residential']: #LTS 2
					newtags = dict(way.tags)
					lts_to_assign = 2
					writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
					newtags['lts'] = str(writeval)
					self.writer.add_way(way.replace(tags=newtags))
					if writeval != 4: self.n_modified_ways += 1
				elif way.tags.get('cycleway') == 'lane' and way.tags.get('highway') in ['primary','secondary']: #LTS 3
					newtags = dict(way.tags)
					lts_to_assign = 3
					writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
					newtags['lts'] = str(writeval)
					self.writer.add_way(way.replace(tags=newtags))
					if writeval != 4: self.n_modified_ways += 1
				elif way.tags.get('cycleway:left') == 'lane' and way.tags.get('highway') in ['primary','secondary']: #LTS 3
					newtags = dict(way.tags)
					lts_to_assign = 3
					writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
					newtags['lts'] = str(writeval)
					self.writer.add_way(way.replace(tags=newtags))
					if writeval != 4: self.n_modified_ways += 1
				elif way.tags.get('cycleway:right') == 'lane' and way.tags.get('highway') in ['primary','secondary']: #LTS 3
					newtags = dict(way.tags)
					lts_to_assign = 3
					writeval = lts_to_assign if lts_to_assign <= self.max_lts else 4
					newtags['lts'] = str(writeval)
					self.writer.add_way(way.replace(tags=newtags))
					if writeval != 4: self.n_modified_ways += 1
				else:
					newtags = dict(way.tags)
					newtags['lts'] = '4'
					self.writer.add_way(way.replace(tags=newtags))


def add_lts_tags(osm_filename: str, out_filename: str, max_lts: int = 3) -> None:
	"""
	Adds tags to OSM data indicating LTS for cyclists. Adds tags for levels 1, 2, 3, to cycle-able roads,
	and adds 4 to everything else.
	If max_lts is provided (as 0, 1, 2, 3, or 4), treats roads with higher LTS than max_lts as un-cycleable (lts 4)
	Args:
		osm_filename: input path
		out_filename: output path
		max_lts: the highest level of LTS tags to add (0, 1, 2, 3, or 4)
	"""
	assert max_lts in [0,1,2,3,4]
	writer = osmium.SimpleWriter(out_filename)
	ltsadder = SimplestLTSAdder(writer, max_lts)
	ltsadder.apply_file(osm_filename)
	print(f'With max_lts {max_lts}, added lts=x to {ltsadder.n_modified_ways}')
